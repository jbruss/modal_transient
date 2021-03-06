/* ---------------------------------------------------------------------
 * Modified by Jonathan B. Russ to read in Abaqus mesh from file and
 * perform a modal transient solution in parallel using SLEPc and PETSc
 * for linear algebra.
 * ---------------------------------------------------------------------
 */
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/index_set.h>

// Shared Triangulation
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/lac/slepc_spectral_transformation.h>

// My Parameters Object
#include "Parameters.h"

// Standard C++:
#include <fstream>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <map>

#define PI 3.14159265

namespace ModalAnalysis
{
	using namespace dealii;

	// ********************** RUNGE KUTTA 4TH ORDER TIME INTEGRATION CLASS **********************
	class RungeKuttaIntegrator
	{
		unsigned int current_step_number;
		unsigned int num_steps;
		const double time_step;
		const double wn;
		const double force_scale_factor, damping;
		std::vector<double> modal_force_values;
		double k1d, k1v, k2d, k2v, k3d, k3v, k4d, k4v, ft, ft_dt, ft_average;
		double prev_disp, prev_veloc;
		void
		initialize_input_force (const std::string force_filename);
	public:
		RungeKuttaIntegrator (const unsigned int n_time_steps, const double dt, const double wn,
								const double force_scale, const double damping, const std::string force_filename);
		void
		update_step (double &new_disp, double &new_veloc, double &new_accel);
	};

	RungeKuttaIntegrator::RungeKuttaIntegrator (const unsigned int n_time_steps, const double dt, const double wn,
													const double force_scale, const double damping, const std::string force_filename) :
			num_steps (n_time_steps), time_step (dt), wn (wn), force_scale_factor (force_scale), damping (damping)
	{
		// This is the only constructor available for this class.
		// The number of time steps, time increment, natural frequency, and force scale factor (mode' * force_vector)
		// 		must be specified.
		initialize_input_force (force_filename);
		current_step_number = 0;
	}

	void
	RungeKuttaIntegrator::initialize_input_force (const std::string force_filename)
	{
		// Read in the force from file force_filename - it must have dt = time_increment in the constructor!
		modal_force_values.resize (num_steps + 1, 0.0);
		std::ifstream force_input_file (force_filename, std::ifstream::in);
		double current_value;
		unsigned int i = 0;
		while ((!force_input_file.eof ()) && (i != num_steps + 1))
		{
			force_input_file >> current_value;
			modal_force_values[i] = force_scale_factor * current_value;
			++i;
		}
		force_input_file.close ();
	}

	void
	RungeKuttaIntegrator::update_step (double &new_disp, double &new_veloc, double &new_accel)
	{
		// new_accel is never used but is kept to maintain a consistent method signature
		// 	with the NewmarkBeta integration class!
		ft = modal_force_values[current_step_number];
		ft_dt = modal_force_values[current_step_number + 1];
		ft_average = (ft + ft_dt) / 2.0;

		k1d = time_step * prev_veloc;
		k1v = time_step * (ft - 2 * damping * wn * prev_veloc - (wn * wn) * prev_disp);

		k2d = time_step * (prev_veloc + 0.5 * k1v);
		k2v = time_step
				* (ft_average - 2 * damping * wn * (prev_veloc + 0.5 * k1v)
						- (wn * wn) * (prev_disp + 0.5 * k1d));

		k3d = time_step * (prev_veloc + 0.5 * k2v);
		k3v = time_step
				* (ft_average - 2 * damping * wn * (prev_veloc + 0.5 * k2v)
						- (wn * wn) * (prev_disp + 0.5 * k2d));

		k4d = time_step * (prev_veloc + k3v);
		k4v = time_step
				* (ft_dt - 2 * damping * wn * (prev_veloc + k3v) - (wn * wn) * (prev_disp + k3d));

		new_disp = prev_disp + (k1d + 2 * k2d + 2 * k3d + k4d) / 6.0;
		new_veloc = prev_veloc + (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0;
		prev_disp = new_disp;
		prev_veloc = new_veloc;
		++current_step_number;
	}

	// ********************** NEWMARK BETA TIME INTEGRATION CLASS **********************

	class NewmarkBetaIntegrator
	{
		const unsigned int num_steps;
		const double time_step, wn;
		unsigned int current_step_number;
		double prev_disp, prev_veloc, prev_accel, nb_gamma, nb_beta, a1, a2, a3;
		const double force_scale_factor;
		std::vector<double> modal_force_values;
		void
		initialize_input_force (const std::string force_filename);
	public:
		NewmarkBetaIntegrator (const unsigned int num_steps, const double time_step, const double wn,
														const double force_scale, const double damping, const std::string force_filename);
		void
		update_step (double &new_disp, double &new_veloc, double &new_accel);
	};

	NewmarkBetaIntegrator::NewmarkBetaIntegrator (const unsigned int num_steps,
																								const double time_step, const double wn,
																								const double force_scale, const double damping,
																								const std::string force_filename) :
			num_steps (num_steps), time_step (time_step), wn (wn), force_scale_factor (force_scale)
	{
		initialize_input_force (force_filename);
		prev_disp = 0.0;
		prev_veloc = 0.0;
		prev_accel = 0.0;
		nb_gamma = 0.5;
		nb_beta = 1.0 / 6.0;
		a1 = 1 / (nb_beta * pow (time_step, 2)) + (nb_gamma * 2 * damping * wn) / (nb_beta * time_step);
		a2 = 1 / (nb_beta * time_step) + (nb_gamma / nb_beta - 1) * (2 * damping * wn);
		a3 = (1 / (2 * nb_beta) - 1) + time_step * 2 * damping * wn * (nb_gamma / (2 * nb_beta) - 1);
		current_step_number = 1;
	}

	void
	NewmarkBetaIntegrator::initialize_input_force (const std::string force_filename)
	{
		// Read in the force from file force_filename - it must have dt = time_increment in the constructor!
		modal_force_values.resize (num_steps + 1, 0.0);
		std::ifstream force_input_file (force_filename, std::ifstream::in);
		double current_value;
		unsigned int i = 0;
		while ((!force_input_file.eof ()) && (i != num_steps + 1))
		{
			force_input_file >> current_value;
			modal_force_values[i] = force_scale_factor * current_value;
			++i;
		}
		force_input_file.close ();
	}

	void
	NewmarkBetaIntegrator::update_step (double &new_disp, double &new_veloc, double &new_accel)
	{
		new_disp = (modal_force_values[current_step_number] + a1 * prev_disp + a2 * prev_veloc + a3 * prev_accel)
				/ (pow (wn, 2) + a1);
		new_veloc = (nb_gamma / (nb_beta * time_step)) * (new_disp - prev_disp)
				+ (1 - nb_gamma / nb_beta) * prev_veloc + time_step * (1 - nb_gamma / (2 * nb_beta)) * prev_accel;
		new_accel = 1 / (nb_beta * pow (time_step, 2)) * (new_disp - prev_disp)
				- (1 / (nb_beta * time_step)) * prev_veloc - (1 / (2 * nb_beta) - 1) * prev_accel;
		prev_disp = new_disp;
		prev_veloc = new_veloc;
		prev_accel = new_accel;
		++current_step_number;
	}

	// ********************** EIGENVALUE PROBLEM CLASS **********************
	template<int dim>
		class EigenvalueProblem
		{
		public:
			EigenvalueProblem (const std::string &prm_file);
			void
			run ();

		private:
			void
			make_grid_and_dofs ();
			void
			assemble_system ();
			void
			assemble_force_vector ();
			unsigned int
			solve_static ();
			unsigned int
			solve_eigen_parallel ();
			void
			output_modes_and_partition () const;
			void
			output_static_solution () const;
			void
			output_time_step_solution (const unsigned int time_step);
			void
			write_Eigen_Restart ();
			void
			read_Eigen_Restart ();
			void
			get_Modal_Acceleration (const unsigned int num_time_steps, const double dt, const double wn,
															const double damping, const double force_scale,
															FullMatrix<double> &modal_displacements,
															const unsigned int local_mode_number);
			void
			compute_transient_solution ();

			parallel::shared::Triangulation<dim> triangulation;
			MPI_Comm mpi_communicator;
			FESystem<dim> fe;
			DoFHandler<dim> dof_handler;
			const unsigned int n_mpi_processes;
			const unsigned int this_mpi_process;

			std::vector<types::global_dof_index> local_dofs_per_process;
			unsigned int n_local_cells = 0; // Gets set in make_grid_and_dofs()
			IndexSet locally_owned_dofs;
			IndexSet locally_relevant_dofs;

			PETScWrappers::MPI::SparseMatrix stiffness_matrix;
			PETScWrappers::MPI::SparseMatrix mass_matrix;
			PETScWrappers::MPI::SparseMatrix dynamic_matrix;

			PETScWrappers::MPI::Vector force_vector, static_solution, current_solution;
			std::vector<PETScWrappers::MPI::Vector> eigenvectors;
			std::vector<double> eigenvalues;

			Parameters params;
			ConditionalOStream pcout;

			TimerOutput computing_timer;
			ConstraintMatrix constraints;
		};

	// Constructor with input parameter file handling
	template<int dim>
		EigenvalueProblem<dim>::EigenvalueProblem (const std::string &prm_file) :
				triangulation (MPI_COMM_WORLD),
				mpi_communicator (MPI_COMM_WORLD),
				fe (FE_Q<dim> (2), dim),
				dof_handler (triangulation),
				n_mpi_processes (Utilities::MPI::n_mpi_processes (mpi_communicator)),
				this_mpi_process (Utilities::MPI::this_mpi_process (mpi_communicator)),
				pcout (std::cout, this_mpi_process == 0),
				computing_timer (mpi_communicator, pcout, TimerOutput::summary, TimerOutput::wall_times)
		{
			// Read in the parameters and print them to the console
			params.initialize(prm_file);
			if (this_mpi_process == 0)
			{
				params.print_parameters();
			}
		}

	template<int dim>
		void
		EigenvalueProblem<dim>::make_grid_and_dofs ()
		{
			TimerOutput::Scope t (computing_timer, "setup");
			GridIn<dim> gridin;
			gridin.attach_triangulation (triangulation);
			std::ifstream f (params.geometry_file);
			gridin.read_abaqus (f);
			dof_handler.distribute_dofs (fe);

			locally_owned_dofs = dof_handler.locally_owned_dofs ();
			DoFTools::extract_locally_relevant_dofs (dof_handler, locally_relevant_dofs);
			n_local_cells = GridTools::count_cells_with_subdomain_association (
					triangulation, triangulation.locally_owned_subdomain ());
			local_dofs_per_process = dof_handler.n_locally_owned_dofs_per_processor ();

			eigenvectors.resize (params.nmodes);
			for (unsigned int i = 0; i < eigenvectors.size (); ++i)
				eigenvectors[i].reinit (locally_owned_dofs, mpi_communicator);

			constraints.clear ();
			constraints.reinit (locally_relevant_dofs);
			if (params.fix_boundary)
			{
				DoFTools::make_zero_boundary_constraints (dof_handler, params.sideset_to_fix, constraints);
			}
			constraints.close ();

			DynamicSparsityPattern dsp (locally_relevant_dofs);
			DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, /*keep constrained dofs*/
																				false);
			SparsityTools::distribute_sparsity_pattern (dsp, local_dofs_per_process, mpi_communicator,
																									locally_relevant_dofs);

			force_vector.reinit (locally_owned_dofs, mpi_communicator);
			static_solution.reinit (locally_owned_dofs, mpi_communicator);
			current_solution.reinit (locally_owned_dofs, mpi_communicator);

			stiffness_matrix.reinit (locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
			mass_matrix.reinit (locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
			dynamic_matrix.reinit (locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);

			eigenvalues.resize (eigenvectors.size ());
		}

// Assemble the global matrices
	template<int dim>
		void
		EigenvalueProblem<dim>::assemble_system ()
		{
			TimerOutput::Scope t (computing_timer, "assembly");
			QGauss<dim> quadrature_formula (3);

			FEValues<dim> fe_values (
					fe, quadrature_formula,
					update_values | update_gradients | update_quadrature_points | update_JxW_values);

			const unsigned int dofs_per_cell = fe.dofs_per_cell;
			const unsigned int n_q_points = quadrature_formula.size ();

			FullMatrix<double> cell_stiffness_matrix (dofs_per_cell, dofs_per_cell);
			FullMatrix<double> cell_mass_matrix (dofs_per_cell, dofs_per_cell);
			FullMatrix<double> cell_dynamic_matrix (dofs_per_cell, dofs_per_cell);

			std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

			const FEValuesExtractors::Vector displacements (0);

			double lambda_value, mu_value, density_value;

			typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active (), endc =
					dof_handler.end ();
			for (; cell != endc; ++cell)
			{
				if (cell->is_locally_owned ())
				{
					lambda_value = params.get_lambda (cell->material_id ());
					mu_value = params.get_mu (cell->material_id ());
					density_value = params.get_density (cell->material_id ());

					fe_values.reinit (cell);
					cell_stiffness_matrix = 0;
					cell_mass_matrix = 0;

					for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
						for (unsigned int i = 0; i < dofs_per_cell; ++i)
						{
							const SymmetricTensor<2, dim> phi_i_symmgrad =
									fe_values[displacements].symmetric_gradient (i, q_point);
							const double phi_i_div = fe_values[displacements].divergence (i, q_point);

							for (unsigned int j = 0; j < dofs_per_cell; ++j)
							{
								const SymmetricTensor<2, dim> phi_j_symmgrad =
										fe_values[displacements].symmetric_gradient (j, q_point);
								const double phi_j_div = fe_values[displacements].divergence (j, q_point);

								cell_stiffness_matrix (i, j) += (phi_i_div * phi_j_div * lambda_value
										+ 2 * (phi_i_symmgrad * phi_j_symmgrad) * mu_value) * fe_values.JxW (q_point);

								cell_mass_matrix (i, j) += (fe_values[displacements].value (i, q_point)
										* fe_values[displacements].value (j, q_point) * density_value)
										* fe_values.JxW (q_point);
							}
						}
					// Now that we have the local matrix contributions, we transfer them
					// into the global objects and take care of zero boundary constraints:
					cell->get_dof_indices (local_dof_indices);
					constraints.distribute_local_to_global (cell_stiffness_matrix, local_dof_indices,
																									stiffness_matrix);

					constraints.distribute_local_to_global (cell_mass_matrix, local_dof_indices, mass_matrix);

					cell_dynamic_matrix = 0;
					for (unsigned int i = 0; i < dofs_per_cell; ++i)
					{
						for (unsigned int j = 0; j < dofs_per_cell; ++j)
						{
							cell_dynamic_matrix (i, j) = cell_stiffness_matrix (i, j)
									- params.shift * cell_mass_matrix (i, j);
						}
					}
					constraints.distribute_local_to_global (cell_dynamic_matrix, local_dof_indices,
																									dynamic_matrix);
				}
			}

			stiffness_matrix.compress (VectorOperation::add);
			mass_matrix.compress (VectorOperation::add);
			dynamic_matrix.compress (VectorOperation::add);

			// Before leaving the function, we calculate spurious eigenvalues,
			// introduced to the system by zero Dirichlet constraints. As
			// discussed in the introduction, the use of Dirichlet boundary
			// conditions coupled with the fact that the degrees of freedom
			// located at the boundary of the domain remain part of the linear
			// system we solve, introduces a number of spurious eigenvalues.
			// Below, we output the interval within which they all lie to
			// ensure that we can ignore them should they show up in our
			// computations.
			double min_spurious_eigenvalue =  std::numeric_limits<double>::max ();
			double max_spurious_eigenvalue = -std::numeric_limits<double>::max ();

			double dynamic_ii, mass_ii, ev;
			std::pair<PETScWrappers::MatrixBase::size_type, PETScWrappers::MatrixBase::size_type> range;
			range = dynamic_matrix.local_range ();
			unsigned int lo_idx = range.first;
			unsigned int hi_idx = range.second;
			for (unsigned int i = lo_idx; i < hi_idx; ++i)
			{
				if (constraints.is_constrained (i))
				{
					dynamic_ii = dynamic_matrix.diag_element (i);
					mass_ii = mass_matrix.diag_element (i);
					if (mass_ii <= 1.0e-14)
					{
						std::cout << "One constrained DOF had mass matrix diagonal value < 1e-14. Skipping it."
								<< std::endl;
						continue;
					}
					ev = dynamic_ii / mass_ii;
					min_spurious_eigenvalue = std::min (min_spurious_eigenvalue, ev);
					max_spurious_eigenvalue = std::max (max_spurious_eigenvalue, ev);
				}
			}
			min_spurious_eigenvalue = Utilities::MPI::min (min_spurious_eigenvalue, mpi_communicator);
			max_spurious_eigenvalue = Utilities::MPI::max (max_spurious_eigenvalue, mpi_communicator);
			pcout << "   Spurious frequencies (Hz) are all in the interval " << "["
					<< sqrt (std::abs (min_spurious_eigenvalue + params.shift)) / (2 * PI) << ", "
					<< sqrt (std::abs (max_spurious_eigenvalue + params.shift)) / (2 * PI) << "]" << std::endl;
		}

	template<int dim>
		void
		EigenvalueProblem<dim>::assemble_force_vector ()
		{
			TimerOutput::Scope t (computing_timer, "assemble_force_vector");
			// Figure out which DoFs to apply the force to using boundary_id 2 in x-direction
			IndexSet dofs_to_apply_force;
			ComponentMask component_mask (3, true);
			component_mask.set (0, true);
			component_mask.set (1, false);
			component_mask.set (2, false);
			std::set<types::boundary_id> bndry_id_set;
			bndry_id_set.insert (params.static_force_sideset);
			DoFTools::extract_boundary_dofs (dof_handler, component_mask, dofs_to_apply_force,
																				bndry_id_set);

			const double dof_force_value = -1.0 * params.static_force_scale / dofs_to_apply_force.n_elements ();
			for (auto current_idx = dofs_to_apply_force.begin ();
					current_idx != dofs_to_apply_force.end (); ++current_idx)
			{
				constraints.distribute_local_to_global (*current_idx, dof_force_value, force_vector);
			}
		}

	template<int dim>
		unsigned int
		EigenvalueProblem<dim>::solve_static ()
		{
			assemble_force_vector ();
			TimerOutput::Scope t (computing_timer, "solve_static");
			SolverControl linear_solver_control (dof_handler.n_dofs (), 1.0e-12, false, false);
			PETScWrappers::SolverCG linear_solver (linear_solver_control, mpi_communicator);

			PETScWrappers::PreconditionBlockJacobi preconditioner (stiffness_matrix);
			linear_solver.solve (stiffness_matrix, static_solution, force_vector, preconditioner);
			return linear_solver_control.last_step ();
		}

	template<int dim>
		unsigned int
		EigenvalueProblem<dim>::solve_eigen_parallel ()
		{
			TimerOutput::Scope t (computing_timer, "solve_eigen_parallel");
			pcout << "   Number of eigenpairs requested: " << eigenvalues.size () << std::endl;

			PETScWrappers::PreconditionBlockJacobi::AdditionalData data;
			PETScWrappers::PreconditionBlockJacobi preconditioner (mpi_communicator, data);
			SolverControl linear_solver_control (dof_handler.n_dofs (), 1.0e-12, false, false);
			PETScWrappers::SolverCG linear_solver (linear_solver_control, mpi_communicator);
			linear_solver.initialize (preconditioner);

			SolverControl solver_control (2000, 1e-10, false, false);
			SLEPcWrappers::SolverKrylovSchur eigensolver (solver_control, mpi_communicator);

			double eigen_shift = std::pow (2.0 * PI * params.shift_frequency, 2.0);
			SLEPcWrappers::TransformationShiftInvert::AdditionalData additional_data (eigen_shift);
			SLEPcWrappers::TransformationShiftInvert shift (mpi_communicator, additional_data);
			shift.set_solver (linear_solver);
			eigensolver.set_transformation (shift);
			eigensolver.set_which_eigenpairs (EPS_SMALLEST_REAL);
			eigensolver.set_problem_type (EPS_GHEP);

			eigensolver.solve (dynamic_matrix, mass_matrix, eigenvalues, eigenvectors,
													eigenvectors.size ());

			if (params.mass_normalize_modes)
			{
				pcout << "   Normalizing eigenvectors with respect to the mass matrix." << std::endl;
				for (unsigned int i = 0; i < eigenvectors.size (); ++i)
					eigenvectors[i] /= sqrt (
							mass_matrix.matrix_scalar_product (eigenvectors[i], eigenvectors[i]));
			}
			else
			{
				pcout << "   Normalizing eigenvectors by l_infinity norm." << std::endl;
				for (unsigned int i = 0; i < eigenvectors.size (); ++i)
					eigenvectors[i] /= eigenvectors[i].linfty_norm ();
			}

			// Finally return the number of iterations it took to converge:
			return solver_control.last_step ();
		}

	template<int dim>
		void
		EigenvalueProblem<dim>::output_modes_and_partition () const
		{
			std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation (
					dim, DataComponentInterpretation::component_is_part_of_vector);

			DataOut<dim> data_out;
			data_out.attach_dof_handler (dof_handler);

			std::vector<Vector<double> > eigenvectors_temp;
			eigenvectors_temp.resize (eigenvectors.size ());
			for (unsigned int i = 0; i < eigenvectors.size (); ++i)
			{
				eigenvectors_temp[i].reinit (dof_handler.n_dofs (), false);
				eigenvectors_temp[i] = eigenvectors[i];
				data_out.add_data_vector (eigenvectors_temp[i],
																	std::string ("Mode_") + Utilities::int_to_string (i + 1),
																	DataOut<dim>::type_dof_data, data_component_interpretation);
			}

			std::vector<types::subdomain_id> partition_int (triangulation.n_active_cells ());
			GridTools::get_subdomain_association (triangulation, partition_int);
			const Vector<double> partitioning (partition_int.begin (), partition_int.end ());
			data_out.add_data_vector (partitioning, "partitioning");

			data_out.build_patches ();
			const std::string filename = ("eigen-" + Utilities::int_to_string (this_mpi_process, 1)
					+ ".vtu");
			std::ofstream output (filename.c_str ());
			data_out.write_vtu (output);

			if (this_mpi_process == 0)
			{
				std::vector<std::string> filenames;
				for (unsigned int i = 0; i < n_mpi_processes; ++i)
					filenames.push_back ("eigen-" + Utilities::int_to_string (i, 1) + ".vtu");
				std::ofstream master_output ("eigen_main.pvtu");
				data_out.write_pvtu_record (master_output, filenames);
			}
		}

	template<int dim>
		void
		EigenvalueProblem<dim>::output_static_solution () const
		{
			std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation (
					dim, DataComponentInterpretation::component_is_part_of_vector);

			DataOut<dim> data_out;
			data_out.attach_dof_handler (dof_handler);

			Vector<double> static_sol;
			static_sol.reinit (dof_handler.n_dofs (), false);
			static_sol = static_solution;
			data_out.add_data_vector (static_sol, "static_solution", DataOut<dim>::type_dof_data,
																data_component_interpretation);

			data_out.build_patches ();
			const std::string filename = (params.statics_output_filename + "-" + Utilities::int_to_string (this_mpi_process, 1)
					+ ".vtu");
			std::ofstream output (filename.c_str ());
			data_out.write_vtu (output);

			if (this_mpi_process == 0)
			{
				std::vector<std::string> filenames;
				for (unsigned int i = 0; i < n_mpi_processes; ++i)
					filenames.push_back (params.statics_output_filename + "-" + Utilities::int_to_string (i, 1) + ".vtu");
				std::ofstream master_output (params.statics_output_filename + "_main.pvtu");
				data_out.write_pvtu_record (master_output, filenames);
			}
		}

	template<int dim>
		void
		EigenvalueProblem<dim>::output_time_step_solution (const unsigned int time_step)
		{
			TimerOutput::Scope t (computing_timer, "output_time_step");

			std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation (
					dim, DataComponentInterpretation::component_is_part_of_vector);

			DataOut<dim> data_out;
			data_out.attach_dof_handler (dof_handler);
			Vector<double> temp_solution;
			temp_solution.reinit (dof_handler.n_dofs (), false);
			temp_solution = current_solution;
			data_out.add_data_vector (temp_solution, "disp", DataOut<dim>::type_dof_data,
																data_component_interpretation);
			data_out.build_patches ();

			const std::string filename = params.transient_output_filename + "-" + Utilities::int_to_string (time_step, 4);

			std::ofstream output (
					(filename + "." + Utilities::int_to_string (this_mpi_process, 2) + ".vtu").c_str ());

			data_out.write_vtu (output);
			output.close ();

			if (this_mpi_process == 0)
			{
				std::vector<std::string> filenames;
				for (unsigned int i = 0; i < n_mpi_processes; ++i)
					filenames.push_back (
							params.transient_output_filename + "-" + Utilities::int_to_string (time_step, 4) + "."
									+ Utilities::int_to_string (i, 2) + ".vtu");
				std::ofstream master_output ((filename + ".pvtu").c_str ());
				data_out.write_pvtu_record (master_output, filenames);
				master_output.close ();
			}
		}

	template<int dim>
		void
		EigenvalueProblem<dim>::write_Eigen_Restart ()
		{
			TimerOutput::Scope t (computing_timer, "write_eigen_restart");
			std::ofstream values_output ("eigenvalues.dat", std::ofstream::out);
			for (unsigned int i = 0; i < eigenvectors.size (); ++i)
			{
				std::string filename = ("Mode" + Utilities::int_to_string (i + 1, 2) + "_"
						+ Utilities::int_to_string (this_mpi_process, 1) + ".dat");
				std::ofstream output (filename.c_str (), std::ofstream::out);
				eigenvectors[i].print (output, 6, true, false);
				if (this_mpi_process == 0)
					values_output << std::setprecision (9) << eigenvalues[i] << std::endl;

				output.close ();
			}
			values_output.close ();
		}

	template<int dim>
		void
		EigenvalueProblem<dim>::read_Eigen_Restart ()
		{
			TimerOutput::Scope t (computing_timer, "read_eigen_restart");
			std::ifstream eigenvalues_input ("eigenvalues.dat", std::ifstream::in);
			double eigenval;
			for (unsigned int i = 0; i < eigenvectors.size (); ++i)
			{
				std::string filename = ("Mode" + Utilities::int_to_string (i + 1, 2) + "_"
						+ Utilities::int_to_string (this_mpi_process, 1) + ".dat");
				std::ifstream eigenvector_input (filename.c_str (), std::ifstream::in);

				std::string str;
				// The first 3 parts of the file are [Proc 1 15806-31739] (<- example)... These are not needed
				eigenvector_input >> str;
				eigenvector_input >> str;
				eigenvector_input >> str;

				// Get the starting DoF and ending DoF that this processor owns
				std::pair<PETScWrappers::VectorBase::size_type, PETScWrappers::VectorBase::size_type> locally_owned_range =
						eigenvectors[i].local_range ();
				unsigned int first_dof = locally_owned_range.first;
				unsigned int last_dof = locally_owned_range.second;
				double current_value;
				unsigned int current_dof = first_dof;
				while (current_dof != last_dof + 1)
				{
					eigenvector_input >> current_value;
					eigenvectors[i][current_dof] = current_value;
					++current_dof;
				}
				eigenvector_input.close ();

				eigenvalues_input >> eigenval;
				eigenvalues[i] = eigenval;
			}
			eigenvalues_input.close ();
		}

	template<int dim>
		void
		EigenvalueProblem<dim>::get_Modal_Acceleration (const unsigned int num_time_steps,
																										const double dt, const double wn,
																										const double damping, const double force_scale,
																										FullMatrix<double> &modal_displacements,
																										const unsigned int local_mode_number)
		{
			TimerOutput::Scope t (computing_timer, "get_modal_accel");

			NewmarkBetaIntegrator integrator (num_time_steps, dt, wn, force_scale, damping, params.force_filename);
			double current_disp, current_veloc, current_accel;
			for (unsigned int step = 1; step < num_time_steps; ++step)
			{
				integrator.update_step (current_disp, current_veloc, current_accel);
				modal_displacements (local_mode_number, step) = current_disp;
			}
		}

	template<int dim>
		void
		EigenvalueProblem<dim>::compute_transient_solution ()
		{
			TimerOutput::Scope t (computing_timer, "transient_solution");

			const unsigned int total_number_of_modes = eigenvalues.size ();
			std::vector<double> modal_force_scale_factors;
			modal_force_scale_factors.resize (total_number_of_modes);
			for (unsigned int mode_number = 0; mode_number < total_number_of_modes; ++mode_number)
			{
				modal_force_scale_factors[mode_number] = eigenvectors[mode_number] * force_vector;
			}
			const unsigned int first_mode_locally_owned = this_mpi_process * total_number_of_modes
					/ n_mpi_processes;
			const unsigned int last_mode_locally_owned = (this_mpi_process + 1) * total_number_of_modes
					/ n_mpi_processes - 1;
			const unsigned int num_modes_locally_owned = last_mode_locally_owned
					- first_mode_locally_owned + 1;
			const double damping = 0.03;
			FullMatrix<double> modal_displacements (num_modes_locally_owned, params.nsteps);

			double wn, modal_force_scale;
			for (unsigned int local_mode_number = 0; local_mode_number < num_modes_locally_owned;
					++local_mode_number)
			{
				wn = sqrt (std::abs (eigenvalues[first_mode_locally_owned + local_mode_number] + params.shift));
				modal_force_scale = modal_force_scale_factors[first_mode_locally_owned + local_mode_number];
				get_Modal_Acceleration (params.nsteps, params.time_step, wn, damping, modal_force_scale / params.static_force_scale,
																modal_displacements, local_mode_number);
			}

			// Create a vector that holds the processor rank that owns each mode (use this for the modal superposition)
			std::vector<unsigned int> owning_processor (total_number_of_modes);
			unsigned int current_processor_first_mode_owned, current_processor_last_mode_owned;
			for (unsigned int processor = 0; processor < n_mpi_processes; ++processor)
			{
				current_processor_first_mode_owned = processor * total_number_of_modes / n_mpi_processes;
				current_processor_last_mode_owned = (processor + 1) * total_number_of_modes
						/ n_mpi_processes - 1;
				for (unsigned int mode_number = 0; mode_number < total_number_of_modes; ++mode_number)
				{
					if (mode_number >= current_processor_first_mode_owned
							&& mode_number <= current_processor_last_mode_owned)
						owning_processor[mode_number] = processor;
				}
			}

			// Print out the modal displacement matrices
			const bool print_modal_displacements = false;
			if (print_modal_displacements)
			{
				std::string modal_disp_file_name = "modal_disp_matrix_"
						+ Utilities::int_to_string (this_mpi_process, 1) + ".dat";
				std::ofstream matrix_output (modal_disp_file_name, std::ofstream::out);
				for (unsigned int mode_number = 0; mode_number < num_modes_locally_owned; ++mode_number)
				{
					for (unsigned int step = 0; step < params.nsteps; ++step)
					{
						matrix_output << modal_displacements[mode_number][step] << " ";
					}
					matrix_output << std::endl;
				}
				matrix_output.close ();
			}
			MPI_Barrier (mpi_communicator);
			pcout << "   Finished Computing Modal Displacements." << std::endl << std::endl;

			// Now we step in time and construct the solution vector, which we output at the required frequency
			double current_modal_displacement[1];
			unsigned int output_step_number = 0;
			for (unsigned int step = 0; step < params.nsteps; step += params.nskip)
			{
				current_solution = 0;
				for (unsigned int mode_number = 0; mode_number < total_number_of_modes; ++mode_number)
				{
					// Figure out who owns the modal displacement value and broadcast it to everyone
					if (this_mpi_process == owning_processor[mode_number])
					{
						current_modal_displacement[0] = modal_displacements[mode_number
								- first_mode_locally_owned][step];
					}
					MPI_Bcast (current_modal_displacement, 1, MPI_DOUBLE, owning_processor[mode_number],
											mpi_communicator);
					current_solution.add (current_modal_displacement[0], eigenvectors[mode_number]);
				}
				// Output the solution vector for this time step
				output_time_step_solution (output_step_number);
				++output_step_number;
			}
		}

	template<int dim>
		void
		EigenvalueProblem<dim>::run ()
		{
			time_t start = time (0);
			pcout << std::endl;
			pcout << "*******************  BEGINNING EXECUTION *******************" << std::endl;
			pcout << std::endl;

			make_grid_and_dofs ();

			pcout << "   Number of active cells:       " << triangulation.n_active_cells () << std::endl
					<< "   Number of degrees of freedom: " << dof_handler.n_dofs () << std::endl;

			// Assemble the system of equations (the global matrices)
			assemble_system ();

			// Either solve the Eigenvalue problem again or read in previously computed eigenpairs
			if (params.write_restart)
			{
				unsigned int n_iterations = solve_eigen_parallel ();
				pcout << "   Eigensolver converged in " << n_iterations << " iterations." << std::endl;

				write_Eigen_Restart ();
			}
			else
			{
				pcout << "   Reading in previously computed eigenvectors." << std::endl;
				read_Eigen_Restart ();
			}
			pcout << std::endl;

			// Print the Modal Frequencies to the console
			if (this_mpi_process == 0)
			{
				for (unsigned int i = 0; i < eigenvalues.size (); ++i)
				{
					printf ("   Mode %d Frequency : %0.3f Hz\n", i + 1,
									sqrt (std::abs (eigenvalues[i] + params.shift)) / (2 * PI));
				}
				std::cout << std::endl;
			}

			// If the user requested a static solution and specified a boundary to fix
			// (required for a unique solution) then run the static solve.
			if (params.compute_static_solution && params.fix_boundary)
			{
				unsigned int n_iterations = solve_static ();
				pcout << "   Static Solver converged in " << n_iterations << " iterations." << std::endl;
				output_static_solution();
			}

			// Output the mode shapes, partition, and static solution (if computed) to a file
			output_modes_and_partition ();

			// Clear the dynamic, mass, and stiffness matrices from memory
			dynamic_matrix.clear ();
			mass_matrix.clear ();
			stiffness_matrix.clear ();

			// If we didn't perform a static solution then we need to assemble the force vector
			if (!params.compute_static_solution)
			{
				assemble_force_vector ();
			}

			/* Now compute the solution of the modal equation of each mode
			 that was computed in the eigensolve and sum the contribution
			 of each mode for every time step. We also output the results
			 in this function. */
			compute_transient_solution ();

			char buf[128];
			time_t end = time (0);
			const double w_time = difftime (end, start) / 60.0;
			sprintf (buf, "(%3.1f Minutes)", w_time);
			pcout << "************** END OF EXECUTION " << buf << " **************" << std::endl;
		}
}

int
main (int argc, char **argv)
{
	try
	{
		using namespace dealii;
		using namespace ModalAnalysis;

		Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);

		{
			deallog.depth_console (0);

			EigenvalueProblem<3> problem (argv[1]);
			problem.run ();
		}
	}
	catch (std::exception &exc)
	{
		std::cerr << std::endl << std::endl << "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Exception on processing: " << std::endl << exc.what () << std::endl << "Aborting!"
				<< std::endl << "----------------------------------------------------" << std::endl;

		return 1;
	}
	catch (...)
	{
		std::cerr << std::endl << std::endl << "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Unknown exception!" << std::endl << "Aborting!" << std::endl
				<< "----------------------------------------------------" << std::endl;
		return 1;
	}
	return 0;
}

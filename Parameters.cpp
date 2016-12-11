/*
 * Parameters.cpp
 *
 *      Author: JBR
 */

#include "Parameters.h"

Parameters::Parameters ()
{
	// Give default values... There must be a subsequent call to initialize()
	nmodes = 10;
	shift = -1.0e6;
	shift_frequency = 0.0;
	mass_normalize_modes = true;
	write_restart = true;

	force_filename = "force.dat";
	force_sideset = 2;
	nsteps = 1;
	time_step = 1.0;
	nskip = 1;
	transient_output_filename = "transient";

	compute_static_solution = false;
	static_force_sideset = 2;
	static_force_scale = 1.0;
	statics_output_filename = "static";

	geometry_file = "mesh.inp";

	wtmass = 0.00259;

	fix_boundary = false;
	sideset_to_fix = 1;
}

Parameters::Parameters (const std::string input_file)
{
	std::vector<int> blocks = declare_parameters();
	parameters.read_input (input_file);
	store_parameters(blocks);
}

Parameters::~Parameters ()
{
	// Nothing important enough to destroy.
}

std::vector<int>
Parameters::declare_parameters ()
{
	blocks_params_file.declare_entry ("list_of_blocks", "1", Patterns::Anything (),
																		"Comma-separated list of blocks in the mesh and input file.");
	blocks_params_file.read_input ("blocks_file.inp");
	std::vector<std::string> blocks_list = Utilities::split_string_list(blocks_params_file.get("list_of_blocks"), ',');
	std::vector<int> blocks  = Utilities::string_to_int (blocks_list);

	parameters.enter_subsection("eigen");
	{
		parameters.declare_entry ("nmodes", "10", Patterns::Integer (1, 1000),
															"The number of modes to compute.");
		parameters.declare_entry ("shift", "-1.0", Patterns::Double (-1.0e8, 0.0),
															"Negative shift value for dynamic matrix in case the object is floating.");
		parameters.declare_entry ("shift_frequency", "1.0", Patterns::Double (0.0, 1.0e7),
															"First modal frequency guess for the shift in the eigen-solver.");
		parameters.declare_entry ("Mass Normalize Modes", "true", Patterns::Bool (),
															"Mass normalize eigenvectors or normalize by l_infinity norm.");
		parameters.declare_entry ("Write Restart", "true", Patterns::Bool (),
															"Write the eigenvectors to a set of restart files.");
	}
	parameters.leave_subsection();

	parameters.enter_subsection("transient");
	{
		parameters.declare_entry ("force_filename", "force.dat", Patterns::Anything (),
																	"File containing the force time history.");
		parameters.declare_entry ("force_sideset", "1", Patterns::Integer (1, 10000000),
																	"Sideset to apply force to.");
		parameters.declare_entry ("nsteps", "1", Patterns::Integer (1, 10000000),
															"The number of time steps to take. Must be less than number of entries in the force file.");
		parameters.declare_entry ("time_step", "1.0", Patterns::Double (1.0e-14, 1.0e10),
															"Size of the time step.");
		parameters.declare_entry ("nskip", "1", Patterns::Integer (1, 10000000),
															"Number of steps to skip between output.");
		parameters.declare_entry ("output_filename", "transient", Patterns::Anything (),
															"Root of the output file name for the transient run.");
	}
	parameters.leave_subsection();

	parameters.enter_subsection("statics");
	{
		parameters.declare_entry ("compute_static_solution", "false", Patterns::Bool (),
																	"Whether to compute the static solution if a boundary is fixed.");
		parameters.declare_entry ("force_sideset", "1", Patterns::Integer (1, 10000000),
																	"Sideset to apply force to.");
		parameters.declare_entry ("force_scale", "1.0", Patterns::Double (-1.0e10, 1.0e10),
															"Scale factor for the force.");
		parameters.declare_entry ("output_filename", "static", Patterns::Anything (),
															"Root of the output file name for the statics run.");
	}
	parameters.leave_subsection();

	parameters.enter_subsection("file");
	{
		parameters.declare_entry ("geometry_file", "mesh.inp", Patterns::Anything (),
															"Specify the mesh filename.");
	}
	parameters.leave_subsection();

	parameters.enter_subsection("parameters");
	{
		parameters.declare_entry ("wtmass", "0.00259", Patterns::Double (1e-14, 1.0e10),
															"Conversion factor for weight density to mass density.");
	}
	parameters.leave_subsection();

	parameters.enter_subsection("boundary");
	{
		parameters.declare_entry ("fix_boundary", "false", Patterns::Bool (),
															"Fix boundary at boundary_id specified.");
		parameters.declare_entry ("sideset_to_fix", "1", Patterns::Integer (1, 10000000),
																	"Sideset to fix.");
	}
	parameters.leave_subsection();

	for (unsigned int block_index = 0; block_index < blocks.size(); ++block_index)
	{
		if (blocks[block_index] >= 1 && blocks[block_index] < 10)
			parameters.enter_subsection("Block " + Utilities::int_to_string(blocks[block_index], 1));
		else if (blocks[block_index] >= 10 && blocks[block_index] < 100)
			parameters.enter_subsection("Block " + Utilities::int_to_string(blocks[block_index], 2));
		else if (blocks[block_index] >= 100 && blocks[block_index] < 1000)
			parameters.enter_subsection("Block " + Utilities::int_to_string(blocks[block_index], 3));
		else
			std::cout << "In Parameters::declare_parameters() parsing is not implemented for blocks with more than 3 digits" << std::endl;
		{
			parameters.declare_entry ("E", "1.0e7", Patterns::Double (1e-10, 1.0e14),
																"Modulus of elasticity.");
			parameters.declare_entry ("nu", "0.33", Patterns::Double (0.0, 0.4999),
																"Poisson's Ratio.");
			parameters.declare_entry ("density", "0.0979", Patterns::Double (1e-10, 1.0e14),
																"Density in lb/in^3 usually with wtmass defined in the parameters subsection.");
		}
		parameters.leave_subsection();
	}

	return blocks;
}

void
Parameters::store_parameters (std::vector<int> blocks)
{
	parameters.enter_subsection("eigen");
	{
		nmodes = parameters.get_integer ("nmodes");
		shift = parameters.get_double ("shift");
		shift_frequency = parameters.get_double ("shift_frequency");
		mass_normalize_modes = parameters.get_bool ("Mass Normalize Modes");
		write_restart = parameters.get_bool ("Write Restart");
	}
	parameters.leave_subsection();

	parameters.enter_subsection("transient");
	{
		force_filename = parameters.get ("force_filename");
		force_sideset = parameters.get_integer ("force_sideset");
		nsteps = parameters.get_integer ("nsteps");
		time_step = parameters.get_double ("time_step");
		nskip = parameters.get_integer ("nskip");
		transient_output_filename = parameters.get ("output_filename");
	}
	parameters.leave_subsection();

	parameters.enter_subsection("statics");
	{
		compute_static_solution = parameters.get_bool ("compute_static_solution");
		static_force_sideset = parameters.get_integer ("force_sideset");
		static_force_scale = parameters.get_double ("force_scale");
		statics_output_filename = parameters.get ("output_filename");
	}
	parameters.leave_subsection();

	parameters.enter_subsection("file");
	{
		geometry_file = parameters.get ("geometry_file");
	}
	parameters.leave_subsection();

	parameters.enter_subsection("parameters");
	{
		wtmass = parameters.get_double ("wtmass");
	}
	parameters.leave_subsection();

	parameters.enter_subsection("boundary");
	{
		fix_boundary = parameters.get_bool ("fix_boundary");
		sideset_to_fix = parameters.get_integer ("sideset_to_fix");
	}
	parameters.leave_subsection();

	for (unsigned int block_index = 0; block_index < blocks.size(); ++block_index)
	{
		if (blocks[block_index] >= 1 && blocks[block_index] < 10)
			parameters.enter_subsection("Block " + Utilities::int_to_string(blocks[block_index], 1));
		else if (blocks[block_index] >= 10 && blocks[block_index] < 100)
			parameters.enter_subsection("Block " + Utilities::int_to_string(blocks[block_index], 2));
		else if (blocks[block_index] >= 100 && blocks[block_index] < 1000)
			parameters.enter_subsection("Block " + Utilities::int_to_string(blocks[block_index], 3));
		else
			std::cout << "In Parameters::declare_parameters() parsing is not implemented for blocks with more than 3 digits" << std::endl;
		{
			double E = parameters.get_double ("E");
			double nu = parameters.get_double ("nu");
			lambda_values[blocks[block_index]] = (nu * E) / ((1 + nu) * (1 - 2 * nu));
			mu_values[blocks[block_index]] = E / (2 * (1 + nu));
			density_values[blocks[block_index]] = wtmass * parameters.get_double ("density");
		}
		parameters.leave_subsection();
	}
}

double
Parameters::get_lambda (const int mat_id)
{
	return lambda_values[mat_id];
}

double
Parameters::get_mu (const int mat_id)
{
	return mu_values[mat_id];
}

double
Parameters::get_density (const int mat_id)
{
	return density_values[mat_id];
}

void
Parameters::print_parameters ()
{
	parameters.print_parameters (std::cout, ParameterHandler::OutputStyle::Text);
}

void
Parameters::initialize(const std::string input_file)
{
	std::vector<int> blocks = declare_parameters();
	parameters.read_input (input_file);
	store_parameters(blocks);
}

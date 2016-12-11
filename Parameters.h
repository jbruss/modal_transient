/*
 * Parameters.h
 *
 *      Author: JBR
 */
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>

using namespace dealii;
class Parameters
{
	ParameterHandler parameters;
	ParameterHandler blocks_params_file;
	std::map<int, double> lambda_values;
	std::map<int, double> mu_values;
	std::map<int, double> density_values;
	std::vector<int>
	declare_parameters ();
	void
	store_parameters (std::vector<int> blocks);
public:
	int nmodes;
	double shift;
	double shift_frequency;
	bool mass_normalize_modes;
	bool write_restart;

	std::string force_filename;
	int force_sideset;
	unsigned int nsteps;
	double time_step;
	int nskip;
	std::string transient_output_filename;

	bool compute_static_solution;
	int static_force_sideset;
	double static_force_scale;
	std::string statics_output_filename;

	std::string geometry_file;

	double wtmass;

	bool fix_boundary;
	int sideset_to_fix;

	Parameters ();
	Parameters (const std::string input_file);
	~Parameters ();
	double
	get_lambda (const int mat_id);
	double
	get_mu (const int mat_id);
	double
	get_density (const int mat_id);
	void
	print_parameters();
	void
	initialize(const std::string input_file);
};

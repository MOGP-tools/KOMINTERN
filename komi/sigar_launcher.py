import Apollo3  # noqa
# try:
#     import mpi4py.MPI as mpi  # noqa
# except ImportError:
#     import warnings
#     warnings.warn("mpi4py is not available", Warning)

# Procedures du schema
from PROCEDURES.AP3_PROCEDURES.FA_calculation import calculation_launcher, CalculationType  # noqa
from PROCEDURES.CalculationOptions import CalculationOptions  # noqa

# Donnees du coeur etudie (classe Data)
from benchmark_2 import Data # noqa

import argparse

## Draft of what the launcher should look like for active sampling

if __name__ == "__main__":

    calculation_options = CalculationOptions(
        flux_solver_list=[CalculationOptions.PIJ_MULTICELL()],
        # flux_solver_list=[CalculationOptions.PIJ_MULTICELL(),CalculationOptions.TDT_MOC(anisotropy=CalculationOptions.AnisotropyENUM.P0c)],
        # flux_solver_list=[CalculationOptions.TDT_MOC(anisotropy=CalculationOptions.AnisotropyENUM.P0c)],
        # flux_solver_list=[CalculationOptions.TDT_MOC()],
        # depletion_options=CalculationOptions.DepletionOptions(extrapolation_order="CONSTANT"),
        multi_level_options=CalculationOptions.MultiLevelOptions(energy_mesh="REL2005"),
        ssh_pij_solver=CalculationOptions.SSH_PIJ_MULTICELL(),
        # ssh_pij_solver=CalculationOptions.SSH_PIJ_EXACT(),
        anisotropy=CalculationOptions.AnisotropyENUM.P3,
        critical_leakage=False,
        ups=False,
        redirect_output=True,
        write_completed_tasks=False,
        export_xs=True,
        rates=False,
        create_mpo=False,
        param_values_in_names=True
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-assemblies",  # name on the parser - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=str,
    )
    parser.add_argument(
        "-archive_paths",
        nargs="*",
        type=str,  # any type/callable can be used here
        default=[],
    )
    parser.add_argument(
        "-ctype",
        nargs=1,
        type=str,  # any type/callable can be used here
        default="REPRISE",
    )
    parser.add_argument(
        "-point",
        nargs="*",
        type=float,  # any type/callable can be used here
    )
    parser.add_argument(
        "-study_tag",
        nargs=1,
        type=str,  # any type/callable can be used here
    )
    parser.add_argument(
        "-subset_tag",
        nargs=1,
        type=str,  # any type/callable can be used here
    )
    argnames = ['assemblies', 'archive_paths', 'ctype', 'point', 'study_tag', 'subset_tag']
    # parse the command line
    args = parser.parse_args()
    arg_values = assemblies, list_of_paths, ctype, point, study_tag, subset_tag = [getattr(args, name) for name in argnames]
    ctype, study_tag, subset_tag = ctype[0], study_tag[0], subset_tag[0]
    ctype = getattr(CalculationType, ctype)
    if ctype == CalculationType.REPRISE_AND_DEPLETION:
        calculation_options.export_xs = False
        bu_it = point  # point ==  [closest_bu, bu] dans ce cas
        values_dict = {'bu':bu_it, 'ssh':bu_it}
        points_list = None
    else:
        values_dict = dict(zip(('bu', 'tf', 'tm_dm', 'br'), point))
        points_list = {'var_names': ('bu', 'tf', 'tm_dm', 'br'), 'points': [[point]]}

    nominal_values = {'tm':565., 'tf':874., 'br':500., 'bu':0.}

    if ctype == CalculationType.DEPLETION:
        data = Data(param_temp=Data.ParamTemp.HOT, param_bu=Data.ParamBu.DEPLETION)
    else:
        data = Data(param_temp=Data.ParamTemp.HOT, param_bu=Data.ParamBu.REPRISE, values_dict=values_dict, nominal_values=nominal_values)

    list_of_assemblies = [getattr(data, name) for name in assemblies]
    ##-----------------------------------------------------------------------------------------------------------------
    calculation_launcher(calculation_options=calculation_options,
                         list_of_assemblies=list_of_assemblies,
                         archive_paths=list_of_paths,
                         calculation_type = ctype,
                         points_list = points_list,
                         study_tag = study_tag,
                         subset_tag=subset_tag,
                         overwrite=True)
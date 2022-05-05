from discrete_optimization.rcpsp.rcpsp_model import (
    MethodBaseRobustification,
    MethodRobustification,
    RCPSPModel,
    UncertainRCPSPModel,
    create_poisson_laws_duration,
    create_poisson_laws_resource,
)
from discrete_optimization.rcpsp.rcpsp_parser import (
    files_available,
    get_data_available,
    parse_file,
)


def uncertain_run():
    # file_path = files_available[0]

    files = get_data_available()
    files = [f for f in files if "j1201_1.sm" in f]  # Single mode RCPSP
    # files = [f for f in files if 'j1010_5.mm' in f]  # Multi mode RCPSP
    file_path = files[0]

    rcpsp_model: RCPSPModel = parse_file(file_path)

    uncertain_model: UncertainRCPSPModel = UncertainRCPSPModel(
        base_rcpsp_model=rcpsp_model,
        poisson_laws=create_poisson_laws_duration(rcpsp_model),
    )
    print(uncertain_model)
    worst_model = uncertain_model.create_rcpsp_model(
        MethodRobustification(
            method_base=MethodBaseRobustification.WORST_CASE, percentile=0
        )
    )
    print("Base model : ", rcpsp_model.mode_details)
    print("Worst case : ", worst_model.mode_details)
    best_model = uncertain_model.create_rcpsp_model(
        MethodRobustification(
            method_base=MethodBaseRobustification.BEST_CASE, percentile=0
        )
    )
    print("Best case : ", best_model.mode_details)

    average_model = uncertain_model.create_rcpsp_model(
        MethodRobustification(
            method_base=MethodBaseRobustification.AVERAGE, percentile=0
        )
    )
    print("average case : ", average_model.mode_details)

    sample_model = uncertain_model.create_rcpsp_model(
        MethodRobustification(
            method_base=MethodBaseRobustification.SAMPLE, percentile=0
        )
    )
    print("sample case : ", sample_model.mode_details)

    percentile_model = uncertain_model.create_rcpsp_model(
        MethodRobustification(
            method_base=MethodBaseRobustification.PERCENTILE, percentile=100
        )
    )
    print("percentile case : ", percentile_model.mode_details)

    def get_durations(model: RCPSPModel):
        return [
            model.mode_details[ac][k]["duration"]
            for ac in model.mode_details
            for k in model.mode_details[ac]
        ]

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(5)
    for model, tag, axis in zip(
        [rcpsp_model, worst_model, best_model, sample_model, percentile_model],
        ["base", "worst", "best", "sample", "percentile"],
        ax,
    ):
        sns.distplot(get_durations(model), ax=axis)
        axis.set_title(tag)
    plt.show()


def uncertain_resources():
    # file_path = files_available[0]

    files = get_data_available()
    files = [f for f in files if "j1201_1.sm" in f]  # Single mode RCPSP
    # files = [f for f in files if 'j1010_5.mm' in f]  # Multi mode RCPSP
    file_path = files[0]
    rcpsp_model: RCPSPModel = parse_file(file_path)
    poisson_laws = create_poisson_laws_resource(rcpsp_model)
    uncertain_model: UncertainRCPSPModel = UncertainRCPSPModel(
        base_rcpsp_model=rcpsp_model, poisson_laws=poisson_laws
    )
    print(uncertain_model)
    worst_model = uncertain_model.create_rcpsp_model(
        MethodRobustification(
            method_base=MethodBaseRobustification.WORST_CASE, percentile=0
        )
    )
    print("Base model : ", rcpsp_model.mode_details)
    print("Worst case : ", worst_model.mode_details)
    best_model = uncertain_model.create_rcpsp_model(
        MethodRobustification(
            method_base=MethodBaseRobustification.BEST_CASE, percentile=0
        )
    )
    print("Best case : ", best_model.mode_details)

    average_model = uncertain_model.create_rcpsp_model(
        MethodRobustification(
            method_base=MethodBaseRobustification.AVERAGE, percentile=0
        )
    )
    print("average case : ", average_model.mode_details)

    sample_model = uncertain_model.create_rcpsp_model(
        MethodRobustification(
            method_base=MethodBaseRobustification.SAMPLE, percentile=0
        )
    )
    print("sample case : ", sample_model.mode_details)

    percentile_model = uncertain_model.create_rcpsp_model(
        MethodRobustification(
            method_base=MethodBaseRobustification.PERCENTILE, percentile=100
        )
    )
    print("percentile case : ", percentile_model.mode_details)

    def get_ressource_consumption(model: RCPSPModel):
        return [
            model.mode_details[ac][k][resource]
            for resource in model.resources_list
            for ac in model.mode_details
            for k in model.mode_details[ac]
        ]

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(5)
    for model, tag, axis in zip(
        [rcpsp_model, worst_model, best_model, sample_model, percentile_model],
        ["base", "worst", "best", "sample", "percentile"],
        ax,
    ):
        sns.distplot(get_ressource_consumption(model), ax=axis)
        axis.set_title(tag)
    plt.show()


if __name__ == "__main__":
    uncertain_resources()

import wandb
import numpy as np

NEPOCHS = 350
LR = 0.001
GAMMA = 5

api = wandb.Api()


def generate_key_from_config(config):
    key = ""
    class_model = config["model"]["class"]
    class_projection = config["model"]["projection"]["class"]
    downsampling = config["model"]["downsampling"]
    upsampling = config["model"]["upsampling"]
    dtype = config["dtype"]

    key += class_model

    key += "_"

    key += downsampling

    key += "_"

    key += upsampling

    key += "_"

    if dtype == "complex64":
        key += dtype
        key += "_"
        if class_projection in ["MLPCtoR", "PolyCtoR"]:
            key += class_projection
            key += "_"
            global_value = (
                "Global" if config["model"]["projection"]["global"] else "Local"
            )
            key += global_value
        else:
            if class_projection == "NoCtoR":
                key += class_projection
                if dtype == "complex64":
                    key += "_"
                    key += config["model"]["projection"]["softmax"]
            else:
                key += class_projection
    elif dtype == "float64":
        key += dtype
        key += "_"
        key += config["data"]["transform"].split(",")[1]
    else:
        raise ValueError(f"Unknown dtype {dtype}")

    return key


def display_runs_per_dataset(collected_runs):
    for key, runs in collected_runs.items():
        print(f"  Configuration: {key}")
        for metric, values in runs.items():
            print(metric, values)


def log_collected_runs_to_tex(collected_runs, dataset):
    table_filename = f"{dataset}_table.tex"
    with open(table_filename, "w") as f:
        # Write the LaTeX table header
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write(
            "Model & Best Loss & OA (\%) & Kappa (\%) & F1 (\%) & Precision (\%) & Recall (\%) & IoU (\%) & Cir. S. (\%) & Std. S. (\%) \\\\\n"
        )
        f.write("\\hline\n")

        for key, runs in collected_runs.items():
            if "complex64" in key:
                dtype = "$\mathbb{C}$"
                key = key.replace("_complex64_", "_")
            elif "float64" in key:
                dtype = "$\mathbb{R}$"
                key = key.replace("_float64_", "_")
            else:
                raise ValueError(f"Unknown dtype in {key}")

            model = key.replace("_", " ")
            model = model.replace("  ", " ")

            best_loss = runs["Best Loss"]
            best_loss_mean = sum(best_loss) / len(best_loss)
            best_loss_std = np.std(best_loss)
            oa = runs["test_overall_accuracy"]
            oa_mean = sum(oa) / len(oa)
            oa_std = np.std(oa)
            kappa = runs["test_kappa_score"]
            kappa_mean = sum(kappa) / len(kappa)
            kappa_std = np.std(kappa)
            f1 = runs["test_macro_f1"]
            f1_mean = sum(f1) / len(f1)
            f1_std = np.std(f1)
            precision = runs["test_macro_precision"]
            precision_mean = sum(precision) / len(precision)
            precision_mean = np.std(precision)
            recall = runs["test_macro_recall"]
            recall_mean = sum(recall) / len(recall)
            recall_std = np.std(recall)
            iou = runs["test_mean_iou"]
            iou_mean = sum(iou) / len(iou)
            iou_std = np.std(iou)
            cir_s = runs["test_circ_consistency"]
            cir_s_mean = sum(cir_s) / len(cir_s)
            cir_s_std = np.std(cir_s)
            std_s = runs["test_std_consistency"]
            std_s_mean = sum(std_s) / len(std_s)
            std_s_std = np.std(std_s)
            # Write the table row
            f.write(
                f"{model} & {dtype} & ${round(best_loss_mean, 2)} \\pm {round(best_loss_std,2)}$ & ${round(oa_mean, 2)} \\pm {round(oa_std,2)}$ & ${round(kappa_mean, 2)} \\pm {round(kappa_std,2)}$ & ${round(f1_mean, 2)} \\pm {round(f1_std,2)}$ & ${round(precision_mean, 2)} \\pm {round(precision_mean,2)}$ & ${round(recall_mean, 2)} \\pm {round(recall_std,2)}$ & ${round(iou_mean, 2)} \\pm {round(iou_std,2)}$ & ${round(cir_s_mean, 2)} \\pm {round(cir_s_std,2)}$ & ${round(std_s_mean, 2)} \\pm {round(std_s_std,2)}$ \\\\\n"
            )
            f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write(f"\\caption{{Results for {dataset}}}\n")
        f.write("\\end{table}\n")


def get_collected_runs(dataset):

    project = api.runs("equivariant_cvnn/" + f"{dataset}")

    collected_runs = {}

    for run in project:

        config = run.config
        summary = run.summary

        if (
            "nepochs" not in run.config
            or "optim" not in run.config
            or "params" not in run.config["optim"]
        ):
            print("Skip missing dictionnary entries")
            print(run.name)
            continue
        if (
            run.config["nepochs"] != NEPOCHS
            or run.config["optim"]["params"]["lr"] != LR
            or run.config["loss"]["gamma"] != GAMMA
        ):
            print("Skip wrong hyperparameters")
            print(run.name)
            continue
        if run.state != "finished":
            print("Skip unfinished run")
            print(run.name)
            continue
        if "Best Loss" not in summary:
            print("Skip missing metrics")
            print(run.name)

            print(summary)
            input()
            continue

        # print(run.config)
        # artifacts = run.logged_artifacts()

        data_config = config["data"]
        model_config = config["model"]
        model_class = config["model"]["class"]

        # Prepare the list to hold all the runs for this
        # dataset and specific configuration
        key = generate_key_from_config(config)
        if key not in collected_runs:
            # collected_runs[dataset][key] = {
            #     "Average Accuracy Score": [],
            #     "Best Loss": [],
            #     "Circular shift consistency": [],
            #     "Jaccard Score": [],
            #     "Kappa Score": [],
            #     "Overall Accuracy Score": [],
            #     "Standard shift consistency": [],
            # }
            collected_runs[key] = {
                "Best Loss": [],
                "test_circ_consistency": [],
                "test_std_consistency": [],
                "test_kappa_score": [],
                "test_macro_f1": [],
                "test_macro_precision": [],
                "test_macro_recall": [],
                "test_mean_iou": [],
                "test_overall_accuracy": [],
            }

        for metric, value in collected_runs[key].items():
            collected_runs[key][metric].append(summary[metric])

    display_runs_per_dataset(collected_runs)
    return collected_runs

    # nepochs = 1250
    # optim.param.lr = 0.005

    # model.class
    # dtype : complex64 ou float64
    # projection.class : MLPCtoR ou autres
    # model.projection.softmax = SoftmaxMeantoR

    # model.projection.global = true ou pas
    # data.transform  = si tu fais RVNN en amplitude ou en splitté
    # et le dataset name

    # et on prends dans Summary metrics


if __name__ == "__main__":
    dataset = "polsf"
    collected_runs = get_collected_runs(dataset)
    log_collected_runs_to_tex(collected_runs, dataset)

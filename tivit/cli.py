import json
import os
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import numpy as np
import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from aeon.datasets.tsc_datasets import multivariate, univariate
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
from momentfm import MOMENTPipeline
from tqdm import tqdm

from tivit.analysis import (
    get_intrinsic_dimension,
    get_principal_components,
    measure_alignment,
)
from tivit.arguments import parse_args
from tivit.classifier import train_classifier
from tivit.datautils import get_dataloader
from tivit.embedding import concat_embeddings, embed
from tivit.tivit import get_tivit
from tivit.utils import get_patch_size, set_random_seed, write_result_table


def main():
    args = parse_args()

    if args.classifier_type and args.aggregation == "all":
        raise ValueError(
            "`--aggregation all` returns token-level representations and cannot be "
            "used with `--classifier_type`. Choose `--aggregation mean` or "
            "`--aggregation cls_token` for classification."
        )

    set_random_seed(args.random_seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    available_models = []

    if args.vit_1_name:
        available_models.append(args.vit_1_name.split("/")[1].replace("-", "_"))
    if args.vit_2_name:
        available_models.append(args.vit_2_name.split("/")[1].replace("-", "_"))
    if args.mantis:
        available_models.append("mantis")
    if args.moment:
        available_models.append(f"moment_{args.moment}")

    available_models = "_".join(available_models)

    result_dir = f"{args.result_dir}/{timestamp}_{args.datasets}_{available_models}_{args.classifier_type}"
    os.makedirs(result_dir, exist_ok=False)

    args_dict = vars(args)
    with open(f"{result_dir}/args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.datasets == "ucr":
        datasets = univariate
    elif args.datasets == "uea":
        datasets = multivariate
    else:
        raise ValueError("Only UCR and UEA benchmark available.")

    for dataset in tqdm(datasets):
        print(dataset)

        train_loader, train_labels, test_loader, test_labels = get_dataloader(
            dataset, args
        )

        print("Samples: ", len(train_loader.dataset) + len(test_loader.dataset))
        channels, T = train_loader.dataset[0][0].shape

        mantis_embedding = None
        moment_embedding = None
        vision_embedding_1 = None
        vision_embedding_2 = None

        if args.mantis:
            network = Mantis8M(device=device)
            network = network.from_pretrained("paris-noah/Mantis-8M")
            network = network.to(device)
            mantis_trainer = MantisTrainer(device=device, network=network)

            mantis_embedding = (
                embed(mantis_trainer, train_loader, "mantis", channels, device),
                embed(mantis_trainer, test_loader, "mantis", channels, device),
            )

        if args.moment:
            moment = MOMENTPipeline.from_pretrained(
                f"AutonLab/MOMENT-1-{args.moment}",
                model_kwargs={"task_name": "embedding"},
            )
            moment.init()
            moment.to(device).float()
            moment.eval()

            moment_embedding = (
                embed(moment, train_loader, "moment", channels, device),
                embed(moment, test_loader, "moment", channels, device),
            )

        patch_sizes = get_patch_size(patch_size=args.patch_size, T=T)

        for p in patch_sizes:
            if p:
                print(f"Patch size: {p}")

            if args.vit_1_name:
                tivit = get_tivit(
                    model_name=args.vit_1_name,
                    model_layer=args.vit_1_layer,
                    aggregation=args.aggregation,
                    stride=args.stride,
                    patch_size=p,
                )
                tivit = tivit.to(device=device)
                tivit.eval()

                vision_embedding_1 = (
                    embed(tivit, train_loader, "tivit", channels, device),
                    embed(tivit, test_loader, "tivit", channels, device),
                )

            if args.vit_2_name:
                tivit = get_tivit(
                    model_name=args.vit_2_name,
                    model_layer=args.vit_2_layer,
                    aggregation=args.aggregation,
                    stride=args.stride,
                    patch_size=p,
                )
                tivit = tivit.to(device=device)
                tivit.eval()

                vision_embedding_2 = (
                    embed(tivit, train_loader, "tivit", channels, device),
                    embed(tivit, test_loader, "tivit", channels, device),
                )

            if args.classifier_type:
                train_embeds, test_embeds = concat_embeddings(
                    vision_embedding_1,
                    vision_embedding_2,
                    mantis_embedding,
                    moment_embedding,
                )

                acc_val, acc_test = train_classifier(
                    train_embeds,
                    train_labels,
                    test_embeds,
                    test_labels,
                    args.classifier_type,
                    args.random_seed,
                )
                print(f"Test accuracy: {acc_test:.2f}")

                write_result_table(
                    result_dir=result_dir,
                    dataset=dataset,
                    acc_val=acc_val,
                    acc_test=acc_test,
                    patch_size=p,
                )
            elif args.measure_alignment:
                measure_alignment(
                    mantis_embedding,
                    moment_embedding,
                    vision_embedding_1,
                    vision_embedding_2,
                    dataset,
                    result_dir,
                )
            elif args.get_intrinsic_dimension or args.get_principal_components:
                embeddings = [
                    e
                    for e in [
                        vision_embedding_1,
                        vision_embedding_2,
                        mantis_embedding,
                        moment_embedding,
                    ]
                    if e is not None
                ]

                assert (
                    len(embeddings) == 1
                ), "Compute intrinsic dimensionality only for one model."

                embedding = np.concatenate(embeddings[0], axis=0).transpose(2, 0, 1)

                if args.get_intrinsic_dimension:
                    get_intrinsic_dimension(embedding, dataset, result_dir)
                if args.get_principal_components:
                    get_principal_components(embedding, dataset, result_dir)
            else:
                raise ValueError(
                    "Please choose: linear probing, intrinsic dimension, principal components, alignment."
                )

            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

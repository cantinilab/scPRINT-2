import os
from typing import Any, Dict, List, Optional

import bionty as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import torch.nn.functional as F
from anndata import AnnData, concat
from scdataloader import Collator, Preprocessor
from scdataloader.utils import get_descendants, random_str
from scib_metrics.benchmark import Benchmarker
from scipy.stats import spearmanr
from simpler_flash import FlashTransformer
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from scprint2.model import loss
from scprint2.tasks._knn_cells import ScanpyNeighborAnnDataset
from scprint2.tasks._model_genes import (
    active_model_organisms,
    collator_for_organism_blocks,
    model_gene_dataframe,
    set_collator_organism_ids,
    validate_collator_gene_offsets,
)

FILE_LOC = os.path.dirname(os.path.realpath(__file__))


def _trainable_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in module.named_parameters()
        if param.requires_grad
    }


def _restore_trainable_state(
    module: torch.nn.Module, state: Dict[str, torch.Tensor]
) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name in state:
                param.copy_(state[name].to(device=param.device, dtype=param.dtype))


def _optimizer_step(
    total_loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    scaler: torch.cuda.amp.GradScaler | None,
) -> None:
    """Backpropagate on both CUDA/AMP and CPU without assuming a scaler."""
    if scaler is None:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()


class FinetuneBatchClass:
    def __init__(
        self,
        batch_key: str = "batch",
        predict_keys: Optional[List[str]] = None,
        max_len: int = 5000,
        learn_batches_on: Optional[str] = None,
        num_workers: int = 8,
        batch_size: int = 16,
        num_epochs: int = 8,
        do_mmd_on: Optional[str] = "cell_type_ontology_term_id",
        lr: float = 0.0002,
        ft_mode: str = "xpressor",
        frac_train: float = 0.8,
        loss_scalers: Optional[dict] = None,
        use_knn: bool = True,
        legacy_detached_mmd: bool = False,
        restore_best_model: bool = True,
        train_organism_decoder: bool = False,
    ):
        """
        Embedder a class to embed and annotate cells using a model

        Args:
            batch_key (str, optional): The key in adata.obs that indicates the batch information. Defaults to "batch".
            learn_batches_on (str, optional): The key in adata.obs to learn batch embeddings on. Defaults to None.
                if none, will not learn the batch embeddings.
                the goal is e.g. when having a new species, to learn an embedding for it during finetuning and replace
                the "learn_batches_on" embedding in the model with it, in this case it should be "organism_ontology_term_id".
                batch correction might indeed be better learnt with this additional argument in some cases.
            do_mmd_on (str, optional): Model token whose representation should lose
                information about the groups stored in ``batch_key``. Defaults to
                ``"cell_type_ontology_term_id"``, so MMD is active by default.
            predict_keys (List[str], optional): List of keys in adata.obs to predict
                during fine-tuning. Defaults to ["cell_type_ontology_term_id"].
            batch_size (int, optional): The size of the batches to be used in the DataLoader. Defaults to 64.
            num_workers (int, optional): The number of worker processes to use for data loading. Defaults to 8.
            max_len (int, optional): The maximum length of the sequences to be processed. Defaults to 5000.
            lr (float, optional): The learning rate for the optimizer. Defaults to 0.0002.
            num_epochs (int, optional): The number of epochs to train the model. Defaults to 8.
            ft_mode (str, optional): The fine-tuning mode, either "xpressor" or "full". Defaults to "xpressor".
            frac_train (float, optional): The fraction of data to be used for training. Defaults to 0.8.
            loss_scalers (dict, optional): A dictionary specifying the scaling factors
                for different loss components. ``mmd`` defaults to ``0.03``; expr,
                class, kl, and any of the predict_keys can also be specified.
            use_knn (bool, optional): Whether to use k-nearest neighbors information. Defaults to True.
            legacy_detached_mmd (bool, optional): Reproduce the historical MMD
                bookkeeping bug: compute the legacy one-vs-rest energy statistic,
                convert it to a Python float, and add it to the displayed loss
                without contributing gradients. Defaults to False.
            restore_best_model (bool, optional): Restore the trainable weights from
                the lowest validation-loss epoch. Disable this to reproduce the
                historical helper, which returned the final epoch. Defaults to True.
            train_organism_decoder (bool, optional): Train the organism classifier
                decoder when organism is an explicit prediction target. The fresh
                task3 run added organism after selecting trainable decoders, so its
                organism decoder remained frozen. Defaults to False for parity.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batch_key = batch_key
        self.learn_batches_on = learn_batches_on
        # Keep supervision targets separate from metadata required by the
        # collator.  In particular, organism_ontology_term_id is needed to map
        # genes but must not silently become an additional classification loss.
        self.predict_keys = list(predict_keys or ["cell_type_ontology_term_id"])
        self.max_len = max_len
        self.lr = lr
        self.num_epochs = num_epochs
        self.ft_mode = ft_mode
        self.frac_train = frac_train
        self.batch_emb = None
        self.batch_encoder = {}
        self.do_mmd_on = do_mmd_on
        self.loss_scalers = {"mmd": 0.03, **dict(loss_scalers or {})}
        self.use_knn = use_knn
        self.legacy_detached_mmd = legacy_detached_mmd
        self.restore_best_model = restore_best_model
        self.train_organism_decoder = train_organism_decoder
        self.batch_group_position: int | None = None

    def __call__(
        self,
        model: torch.nn.Module,
        adata: AnnData = None,
        train_data: AnnData = None,
        val_data: AnnData = None,
    ) -> torch.nn.Module:
        """
        __call__ function to call the embedding

        Args:
            model (torch.nn.Module): The scPRINT model to be used for embedding and annotation.
            adata (AnnData): The annotated data matrix of shape n_obs x n_vars. Rows correspond to cells and columns to genes.
                Defaults to None.
                if provided, it will be split into training and validation sets.
            train_data (AnnData, optional): The training data. Defaults to None.
                if adata is provided, this will be ignored.
            val_data (AnnData, optional): The validation data. Defaults to None.
                if adata is provided, this will be ignored.

        Raises:
            ValueError: If the model does not have a logger attribute.
            ValueError: If the model does not have a global_step attribute.

        Returns:
            torch.nn.Module: the fine-tuned model
        """
        # one of "all" "sample" "none"
        model.predict_mode = "none"
        if self.ft_mode == "xpressor":
            for val in model.parameters():
                val.requires_grad = False
                # setting all to TRUE

            for val in model.cell_transformer.parameters():
                val.requires_grad = True
            for val in model.transformer.blocks[-1].parameters():
                val.requires_grad = True
            for block in model.transformer.blocks:
                for parameter in block.cross_attn.parameters():
                    parameter.requires_grad = True
            for val in model.compressor.parameters():
                val.requires_grad = True
            for class_name in self.predict_keys:
                if (
                    class_name == "organism_ontology_term_id"
                    and not self.train_organism_decoder
                ):
                    continue
                for parameter in model.cls_decoders[class_name].parameters():
                    parameter.requires_grad = True
        elif self.ft_mode == "full":
            for val in model.parameters():
                val.requires_grad = True
        else:
            raise ValueError("ft_mode must be one of 'xpressor' or 'full'")

        # PREPARING THE DATA
        if adata is not None:
            n_train = int(self.frac_train * len(adata))
            train_idx = np.random.choice(len(adata), n_train, replace=False)
            val_idx = np.setdiff1d(np.arange(len(adata)), train_idx)
            if self.do_mmd_on is not None:
                # The inputs can be concatenated in batch blocks. Keep a fixed
                # randomized validation order so every
                # validation batch can measure MMD across the user-defined
                # ``batch_key`` groups instead of returning zero for one group.
                np.random.shuffle(val_idx)

            train_data = adata[train_idx].copy()
            val_data = adata[val_idx].copy()

            print(f"Training data: {train_data.shape}")
            print(f"Validation data: {val_data.shape}")

        mencoders = {}
        for k, v in model.label_decoders.items():
            mencoders[k] = {va: ke for ke, va in v.items()}
        # this needs to remain its original name as it is expect like that by collator, otherwise need to send org_to_id as params

        for i in self.predict_keys:
            if len(set(train_data.obs[i]) - set(mencoders[i].keys())) > 0:
                print("missing labels for ", i)
                train_data.obs[i] = train_data.obs[i].apply(
                    lambda x: x if x in mencoders[i] else "unknown"
                )
        organism_key = "organism_ontology_term_id"
        required_obs = set(self.predict_keys + [self.batch_key, organism_key])
        missing_obs = required_obs - set(train_data.obs)
        if missing_obs:
            raise KeyError(
                "Fine-tuning data is missing required obs columns: "
                + ", ".join(sorted(missing_obs))
            )
        dataset_obs_keys = list(
            dict.fromkeys(self.predict_keys + [self.batch_key, organism_key])
        )
        class_names = list(dict.fromkeys(self.predict_keys + [self.batch_key]))
        self.batch_group_position = class_names.index(self.batch_key)
        if self.do_mmd_on is not None:
            train_groups = train_data.obs[self.batch_key].nunique(dropna=True)
            if train_groups < 2:
                raise ValueError(
                    "MMD requires at least two values in "
                    f"obs[{self.batch_key!r}]; found {train_groups}"
                )
            if val_data is not None:
                validation_groups = val_data.obs[self.batch_key].nunique(dropna=True)
                if validation_groups < 2:
                    raise ValueError(
                        "MMD validation requires at least two values in "
                        f"obs[{self.batch_key!r}]; found {validation_groups}"
                    )
        use_metacell_knn = model.expr_emb_style == "metacell" and self.use_knn
        if use_metacell_knn:
            import scanpy as sc

            for split_data in (train_data, val_data):
                if split_data is None:
                    continue
                if "X_pca" not in split_data.obsm:
                    raise ValueError(
                        "KNN fine-tuning requires adata.obsm['X_pca'] from the "
                        "CP10K/log1p expression preprocessing"
                    )
                sc.pp.neighbors(
                    split_data,
                    n_neighbors=min(15, split_data.n_obs - 1),
                    use_rep="X_pca",
                    random_state=42,
                )

        # create datasets
        self.batch_encoder = {
            i: n
            for n, i in enumerate(
                train_data.obs[self.batch_key].astype("category").cat.categories
            )
        }
        mencoders[self.batch_key] = self.batch_encoder
        train_dataset = ScanpyNeighborAnnDataset(
            train_data,
            obs_to_output=dataset_obs_keys,
            get_knn_cells=use_metacell_knn,
            encoder=mencoders,
        )
        if val_data is not None:
            for i in self.predict_keys:
                if len(set(val_data.obs[i]) - set(mencoders[i].keys())) > 0:
                    val_data.obs[i] = val_data.obs[i].apply(
                        lambda x: x if x in mencoders[i] else "unknown"
                    )
            self.batch_encoder.update(
                {
                    i: n + len(self.batch_encoder)
                    for n, i in enumerate(
                        val_data.obs[self.batch_key].astype("category").cat.categories
                    )
                    if i not in self.batch_encoder
                }
            )
            mencoders[self.batch_key] = self.batch_encoder
            if self.learn_batches_on is not None:
                unseen_validation_batches = set(val_data.obs[self.batch_key]) - set(
                    train_data.obs[self.batch_key]
                )
                if unseen_validation_batches:
                    raise ValueError(
                        "Validation contains batch values absent from training: "
                        + ", ".join(sorted(map(str, unseen_validation_batches)))
                    )
            val_dataset = ScanpyNeighborAnnDataset(
                val_data,
                obs_to_output=dataset_obs_keys,
                get_knn_cells=use_metacell_knn,
                encoder=mencoders,
            )

        # Create collator
        active_organisms = active_model_organisms(model, train_data.obs)
        collator = Collator(
            organisms=active_organisms,
            valid_genes=model.genes,
            class_names=class_names,
            how="random expr",  # or "all expr" for full expression
            max_len=self.max_len,
            org_to_id=mencoders.get("organism_ontology_term_id", {}),
            genedf=model_gene_dataframe(model, train_data.var),
        )
        set_collator_organism_ids(
            collator,
            active_organisms,
            mencoders.get("organism_ontology_term_id", {}),
        )
        validate_collator_gene_offsets(
            collator,
            model,
            active_organisms,
            mencoders.get("organism_ontology_term_id", {}),
        )
        collate_fn = collator_for_organism_blocks(
            collator,
            train_data.var,
            active_organisms,
            mencoders.get("organism_ontology_term_id", {}),
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,  # Adjust based on GPU memory
            num_workers=self.num_workers,
            shuffle=True,
        )
        if val_data is not None:
            val_loader = DataLoader(
                val_dataset,
                collate_fn=collate_fn,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )

        if self.learn_batches_on is not None:
            if val_data is not None:
                print(
                    "all batch key values in val_data should also be present in train_adata!!!"
                )
            self.batch_emb = torch.nn.Embedding(
                num_embeddings=train_data.obs[self.batch_key].nunique(),
                # This embedding replaces output_cell_embs before compression,
                # so it must use the transformer width rather than the latent
                # compressor width (8 in small-v2 versus d_model=256).
                embedding_dim=model.d_model,
            ).to(model.device)

        ## PREPARING THE OPTIM
        all_params = (
            list(model.parameters())
            # + list(batch_cls.parameters())
            + (
                list(self.batch_emb.parameters())
                if self.learn_batches_on is not None
                else []
            )
        )

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            all_params,
            lr=self.lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Setup scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        # Setup automatic mixed precision
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        for k, i in model.mat_labels_hierarchy.items():
            model.mat_labels_hierarchy[k] = i.to(model.device)

        ## train
        best_val_loss = None
        best_epoch = None
        best_model_state = None
        best_batch_emb_state = None
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")

            # Training phase
            train_loss = 0.0
            train_steps = 0
            avg_expr = 0
            avg_cls = 0
            avg_mmd = 0

            pbar = tqdm(train_loader, desc="Training")
            model.train()
            for batch_idx, batch in enumerate(pbar):
                optimizer.zero_grad()
                total_loss, cls_loss, mmd, loss_expr = self.batch_corr_pass(
                    batch, model
                )
                _optimizer_step(total_loss, optimizer, model, scaler)

                train_loss += total_loss.item()
                train_steps += 1
                avg_cls += cls_loss.item()
                avg_expr += loss_expr.item()
                avg_mmd += mmd
                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{total_loss.item():.4f}",
                        "avg_loss": f"{train_loss / train_steps:.4f}",
                        "cls_loss": f"{cls_loss.item():.4f}",
                        "mmd_loss": f"{mmd:.4f}",
                        "expr_loss": f"{loss_expr.item():.4f}",
                    }
                )

            # Validation phase
            if val_data is not None:
                model.eval()
                val_loss = 0.0
                val_steps = 0
                val_loss_expr = 0.0
                val_mmd = 0.0
                val_cls = 0.0
                val_loss_to_prt = 0.0

                with torch.no_grad():
                    for batch in val_loader:  # tqdm(val_loader, desc="Validation"):
                        loss_val, cls_loss, mmd, loss_expr = self.batch_corr_pass(
                            batch, model
                        )
                        val_loss_to_prt += loss_val.item()
                        val_loss += loss_val.item()
                        val_steps += 1
                        val_loss_expr += loss_expr.item()
                        val_mmd += mmd
                        val_cls += cls_loss.item()
                try:
                    avg_val_loss = val_loss_to_prt / val_steps
                    avg_train_loss = train_loss / train_steps
                except ZeroDivisionError:
                    print(
                        "Error: Division by zero occurred while calculating average losses."
                    )
                    avg_train_loss = 0
                print(
                    "cls_loss: {:.4f}, mmd_loss: {:.4f}, expr_loss: {:.4f}".format(
                        val_cls / val_steps,
                        val_mmd / val_steps,
                        val_loss_expr / val_steps,
                    )
                )
                print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

                if best_val_loss is None or avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    best_model_state = _trainable_state_dict(model)
                    best_batch_emb_state = (
                        _trainable_state_dict(self.batch_emb)
                        if self.learn_batches_on is not None
                        else None
                    )

                # Store LR before scheduler step for comparison
                lr_before = optimizer.param_groups[0]["lr"]

                # Update learning rate
                scheduler.step(avg_val_loss)

                # Check if LR was reduced
                lr_after = optimizer.param_groups[0]["lr"]
                if lr_after < lr_before:
                    print(
                        f"🔻 Learning rate reduced from {lr_before:.2e} to {lr_after:.2e} (factor: {lr_after / lr_before:.3f})"
                    )
                else:
                    print(f"✅ Learning rate unchanged: {lr_after:.2e}")

                # Early stopping check (simple implementation)
                if epoch > 3 and val_loss / val_steps > 1.3 * avg_train_loss:
                    print("Early stopping due to overfitting")
                    break

        if self.restore_best_model and best_model_state is not None:
            _restore_trainable_state(model, best_model_state)
            if best_batch_emb_state is not None:
                _restore_trainable_state(self.batch_emb, best_batch_emb_state)
            print(
                "Restored best validation trainable weights "
                f"(val loss: {best_val_loss:.4f})"
            )
        self.best_epoch = best_epoch
        self.best_val_loss = best_val_loss

        print("Manual fine-tuning completed!")
        model.eval()
        return model

    def batch_corr_pass(self, batch, model):
        gene_pos = batch["genes"].to(model.device)
        expression = batch["x"].to(model.device)
        depth = batch["depth"].to(model.device)
        class_elem = batch["class"].long().to(model.device)
        total_loss = 0
        # Forward pass with automatic mixed precisio^n
        with torch.cuda.amp.autocast():
            # Forward pass
            output = model.forward(
                gene_pos,
                expression,
                neighbors=(
                    batch["knn_cells"].to(model.device)
                    if model.expr_emb_style == "metacell" and self.use_knn
                    else None
                ),
                neighbors_info=(
                    batch["knn_cells_info"].to(model.device)
                    if model.expr_emb_style == "metacell" and self.use_knn
                    else None
                ),
                req_depth=depth,
                depth_mult=expression.sum(1),
                do_class=True,
                metacell_token=torch.zeros_like(depth),
            )
            ## adaptor on ct_emb
            # ctpos = model.classes.index("cell_type_ontology_term_id") + 1
            # emb = output["output_cell_embs"][:, ctpos, :]
            #
            # output["output_cell_embs"][:, ctpos, :] = adaptor_layer(
            #    torch.cat([emb, class_elem[:, 1].unsqueeze(1).float()], dim=1)
            # )
            if self.learn_batches_on is not None:
                if self.batch_group_position is None:
                    raise RuntimeError("batch group position was not initialized")
                batch_pos = model.classes.index(self.learn_batches_on) + 1
                output["output_cell_embs"][:, batch_pos, :] = self.batch_emb(
                    class_elem[:, self.batch_group_position]
                )

            ## generate expr loss
            output_gen = model._generate(
                cell_embs=output["output_cell_embs"],
                gene_pos=gene_pos,
                depth_mult=expression.sum(1),
                req_depth=depth,
            )
            if "zero_logits" in output_gen:
                loss_expr = loss.zinb(
                    theta=output_gen["disp"],
                    pi=output_gen["zero_logits"],
                    mu=output_gen["mean"],
                    target=expression,
                )
                if model.zinb_and_mse:
                    loss_expr += (
                        loss.mse(
                            input=torch.log(output_gen["mean"] + 1)
                            * (1 - torch.sigmoid(output_gen["zero_logits"])),
                            target=torch.log(expression + 1),
                        )
                        / 10  # scale to make it more similar to the zinb
                    )
            else:
                loss_expr = loss.mse(
                    input=torch.log(output_gen["mean"] + 1),
                    target=torch.log(expression + 1),
                )
            # Add expression loss to total
            total_loss += loss_expr * self.loss_scalers.get("expr", 0.5)

            # ct
            cls_loss = 0
            for clas in self.predict_keys:
                cls_output = output.get("cls_output_" + clas)
                # ct_output = output["output_cell_embs"][:, ctpos, :]
                # cls_output = model.cls_decoders["cell_type_ontology_term_id"](ct_output)
                cls_loss += loss.hierarchical_classification(
                    pred=cls_output,
                    cl=class_elem[:, self.predict_keys.index(clas)],
                    labels_hierarchy=(
                        model.mat_labels_hierarchy.get(clas).to(model.device)
                        if clas in model.mat_labels_hierarchy
                        else None
                    ),
                ) * self.loss_scalers.get(clas, 1)

            # organ class
            # org_emb = output["compressed_cell_embs"][
            #    model.classes.index("organism_ontology_term_id") + 1
            # ]
            # cls_loss += F.cross_entropy(
            #    input=batch_cls(org_emb),
            #    target=class_elem[:, 1],
            # )
            total_loss += cls_loss * self.loss_scalers.get("class", 1)
            tot_mmd_loss = output["output_cell_embs"].new_zeros(())
            mmd_for_logging = 0.0
            if self.do_mmd_on is not None:
                pos = model.classes.index(self.do_mmd_on) + 1
                # Minimize the positive pairwise distance directly. A gradient
                # reversal here would optimize the MMD in the wrong direction.
                selected_emb = (
                    output["compressed_cell_embs"][pos]
                    if model.compressor is not None
                    else output["input_cell_embs"][:, pos, :]
                )
                if self.batch_group_position is None:
                    raise RuntimeError("Batch group position was not initialized")
                mmd_groups = class_elem[:, self.batch_group_position]
                if self.legacy_detached_mmd:
                    mmd_for_logging = _legacy_detached_one_vs_rest_mmd(
                        selected_emb, mmd_groups
                    )
                    total_loss += mmd_for_logging * self.loss_scalers.get("mmd", 0.03)
                else:
                    # Keep the MMD accumulator attached to the autograd graph.
                    tot_mmd_loss = _balanced_group_pairwise_mmd(
                        selected_emb, mmd_groups
                    )
                    total_loss += tot_mmd_loss * self.loss_scalers.get("mmd", 0.03)
                    mmd_for_logging = tot_mmd_loss.detach().item()
            if "vae_kl_loss" in output:
                total_loss += output["vae_kl_loss"] * self.loss_scalers.get("kl", 0.5)
        return total_loss, cls_loss, mmd_for_logging, loss_expr


def _legacy_detached_one_vs_rest_mmd(
    selected_emb: torch.Tensor, groups: torch.Tensor
) -> float:
    """Return the pre-fix MMD scalar, deliberately detached from autograd."""
    total = 0.0
    for group_id in torch.unique(groups):
        in_group = groups == group_id
        if in_group.sum() < 2 or (~in_group).sum() < 2:
            continue
        mmd = _legacy_unbiased_energy_mmd(
            selected_emb[in_group], selected_emb[~in_group]
        )
        if torch.isnan(mmd):
            print("mmd nan")
        else:
            total += mmd.detach().item()
    return total


def _legacy_unbiased_energy_mmd(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """Historical unbiased energy-distance statistic used before the MMD fix."""
    left_dist = -torch.cdist(left, left, p=2)
    right_dist = -torch.cdist(right, right, p=2)
    cross_dist = -torch.cdist(left, right, p=2)
    n_left = left.shape[0]
    n_right = right.shape[0]
    left_term = left_dist.sum() / (n_left * (n_left - 1)) if n_left > 1 else 0.0
    right_term = right_dist.sum() / (n_right * (n_right - 1)) if n_right > 1 else 0.0
    return left_term + right_term - 2 * cross_dist.mean()


def _balanced_group_pairwise_mmd(
    selected_emb: torch.Tensor, groups: torch.Tensor
) -> torch.Tensor:
    group_ids = torch.unique(groups)
    if group_ids.numel() < 2:
        return selected_emb.new_zeros(())

    valid_mmd_terms = []
    for left_idx in range(group_ids.numel() - 1):
        for right_idx in range(left_idx + 1, group_ids.numel()):
            left_group = groups == group_ids[left_idx]
            right_group = groups == group_ids[right_idx]
            mmd = mmd_loss(selected_emb[left_group], selected_emb[right_group])
            if torch.isnan(mmd):
                print("mmd nan")
            else:
                valid_mmd_terms.append(mmd)

    if valid_mmd_terms:
        return torch.stack(valid_mmd_terms).mean()
    return selected_emb.new_zeros(())


# Compatibility alias for the short-lived task3-specific helper name.
_balanced_species_pairwise_mmd = _balanced_group_pairwise_mmd


def mmd_loss(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute the notebook's unbiased empirical energy-distance statistic.

    Args:
        X (torch.Tensor): Tensor of shape (n1, emb_dim) - first set of embeddings
        Y (torch.Tensor): Tensor of shape (n2, emb_dim) - second set of embeddings

    Returns:
        torch.Tensor: differentiable energy-distance estimate
    """
    return _legacy_unbiased_energy_mmd(X, Y)


class FinetuneGRN:
    pass


class FinetuneGeneEmb:
    pass


class FinetuneNewClass:
    pass


class FinetuneUpdateClass:
    pass

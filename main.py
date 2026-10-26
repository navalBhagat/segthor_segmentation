#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import json
import random
import warnings
from functools import partial
from pathlib import Path
from pprint import pprint
from shutil import copytree, move, rmtree
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, RandomSampler

from dataset import SliceDataset
from ENet import ENet
from losses import LOSS_NAMES, CrossEntropy, get_loss
from metrics_3d import PatientDice, dice_3d, mean_foreground_dice
import optimizers
from ShallowNet import shallowCNN
from stitch import stitch_predictions
from utils import (Dcm,
                   class2one_hot,
                   dice_coef,
                   probs2class,
                   probs2one_hot,
                   save_images,
                   tqdm_)


datasets_params: dict[str, dict[str, Any]] = {
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
    "TOY2": {'K': 2, 'B': 2, 'kernels': 8, 'factor': 2},
    "SEGTHOR": {'K': 5, 'B': 8, 'kernels': 8, 'factor': 2},
    "SEGTHOR_OLD": {'K': 5, 'B': 8, 'kernels': 8, 'factor': 2},
}


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def write_json(path: Path, value) -> None:
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def img_transform(img):
        img = img.convert('L')
        img = np.array(img)[np.newaxis, ...]
        img = img / 255  # max <= 1
        img = torch.tensor(img, dtype=torch.float32)
        return img

def gt_transform(K, img):
        img = np.array(img)[...]
        # The idea is that the classes are mapped to {0, 255} for binary cases
        # {0, 85, 170, 255} for 4 classes
        # {0, 51, 102, 153, 204, 255} for 6 classes
        # Very sketchy but that works here and that simplifies visualization
        img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
        img = torch.tensor(img, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
        img = class2one_hot(img, K=K)
        return img[0]


def dataset_root(dataset: str) -> Path:
    return Path("data") / dataset


def source_data_dir(root_dir: Path) -> Path:
    preprocess_path = root_dir / "preprocess.json"
    if preprocess_path.exists():
        metadata = json.loads(preprocess_path.read_text())
        return Path(metadata["source_dir"])
    return Path("data") / "segthor_train"


def source_scan_pattern(root_dir: Path) -> str:
    return str(source_data_dir(root_dir) / "train" / "{id_}" / "GT.nii.gz")


def validation_gt_path(gt_root: Path, patient_id_: str) -> Path:
    flat_path = gt_root / f"{patient_id_}.nii.gz"
    if flat_path.exists():
        return flat_path

    nested_path = gt_root / patient_id_ / "GT.nii.gz"
    if nested_path.exists():
        return nested_path

    raise FileNotFoundError(f"Could not find {flat_path} or {nested_path}")


def save_3d_validation_metrics(metrics_root: Path,
                               epoch: int,
                               mean_per_class: np.ndarray,
                               foreground_mean: float,
                               per_patient: dict[str, np.ndarray]) -> None:
    np.savez(metrics_root / "dice_3d.npz", **per_patient)
    np.save(metrics_root / "dice_3d_mean_per_class.npy", mean_per_class)
    (metrics_root / "dice_3d_summary.txt").write_text(
        f"epoch={epoch}\nforeground_mean={foreground_mean:.6f}\n"
        f"mean_per_class={mean_per_class.tolist()}\n"
    )


def cleanup_tmp_3d_validation(pred_volumes: Path) -> None:
    tmp_epoch_root = pred_volumes.parent
    tmp_root = tmp_epoch_root.parent
    rmtree(tmp_epoch_root)
    try:
        tmp_root.rmdir()
    except OSError:
        pass


def run_3d_validation(args,
                      epoch: int,
                      K: int,
                      keep_volumes: bool,
                      cleanup: bool = True) -> tuple[np.ndarray, float, dict[str, np.ndarray], Path]:
    root_dir = args.data_root or dataset_root(args.dataset)
    gt_root = args.val_3d_gt_root or source_data_dir(root_dir) / "train"
    preprocess_metadata = root_dir / "preprocess.json"
    preprocess_metadata = preprocess_metadata if preprocess_metadata.exists() else None

    if not gt_root.exists():
        raise FileNotFoundError(f"Missing 3D validation GT folder: {gt_root}")

    pred_slices = args.dest / f"iter{epoch:03d}" / "val"
    if keep_volumes:
        pred_volumes = args.dest / "volumes" / "best_epoch" / "val"
    else:
        pred_volumes = args.dest / "_tmp_3d_val" / f"iter{epoch:03d}" / "val"

    if pred_volumes.exists():
        rmtree(pred_volumes)

    stitch_predictions(data_folder=pred_slices,
                       dest_folder=pred_volumes,
                       source_scan_pattern=source_scan_pattern(root_dir),
                       grp_regex=r"(Patient_\d\d)_\d\d\d\d",
                       num_classes=K,
                       preprocess_metadata_path=preprocess_metadata)

    per_patient: dict[str, np.ndarray] = {}
    for pred_path in sorted(pred_volumes.glob("Patient_*.nii.gz")):
        gt_path = validation_gt_path(gt_root, pred_path.stem[:-4])

        pred_np = np.asarray(nib.load(pred_path).dataobj, dtype=np.uint8)
        gt_np = np.asarray(nib.load(gt_path).dataobj, dtype=np.uint8)
        assert pred_np.shape == gt_np.shape, f"{pred_path.name}: pred {pred_np.shape}, gt {gt_np.shape}"
        patient = pred_path.name.removesuffix(".nii.gz")
        per_patient[patient] = dice_3d(pred_np, gt_np, K=K)

    patient_dice = np.stack(list(per_patient.values()))
    mean_per_class = np.nanmean(patient_dice, axis=0)
    # Each stitched volume contributes one row, so this is already patient-equal.
    foreground_mean = mean_foreground_dice(mean_per_class)

    if keep_volumes:
        save_3d_validation_metrics(pred_volumes.parent,
                                   epoch,
                                   mean_per_class,
                                   foreground_mean,
                                   per_patient)
    elif cleanup:
        cleanup_tmp_3d_validation(pred_volumes)

    return mean_per_class, foreground_mean, per_patient, pred_volumes


def default_net(dataset: str) -> str:
    return 'shallow' if dataset == 'TOY2' else 'enet'


def input_slices(args) -> int:
    return args.slices if args.net == 'enet25d' else 1


def build_model(args, K: int) -> nn.Module:
    if args.net == 'shallow':
        net = shallowCNN(1, K)
    else:
        params = datasets_params[args.dataset]
        net = ENet(input_slices(args), K, kernels=params['kernels'], factor=params['factor'])
    net.init_weights()
    return net


def make_loss(args, K: int):
    if args.mode == "partial" and args.dataset.startswith('SEGTHOR'):
        return CrossEntropy(idk=[0, 1, 3, 4])
    if args.mode != "full":
        raise ValueError((args.mode, args.dataset))

    idk = list(range(K))
    return get_loss(args.loss, idk=idk, **args.loss_kwargs)


def setup(args) -> tuple[nn.Module, Any, torch.device, DataLoader, DataLoader, int]:
    data_generator = seed_everything(args.seed)
    if args.gpu and not torch.cuda.is_available():
        raise RuntimeError('A GPU run was requested but CUDA is unavailable')
    device = torch.device("cuda") if args.gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K = datasets_params[args.dataset]['K']
    net = build_model(args, K).to(device)
    optimizer = optimizers.get_optimizer(net=net,
                                         optimizer_name=args.optimizer,
                                         lr=0.0005,
                                         kwargs=args.optimizer_kwargs)

    batch_size = args.batch_size or datasets_params[args.dataset]['B']
    root_dir = args.data_root or dataset_root(args.dataset)
    common_settings = {
        'img_transform': img_transform,
        'gt_transform': partial(gt_transform, K),
        'debug': args.debug,
        'input_slices': input_slices(args),
        'slice_step': args.slice_step,
        'num_classes': K,
    }
    train_set = SliceDataset('train', root_dir, **common_settings,
                             augmentation_policy=args.augmentation,
                             augmentation_components=args.augmentation_components)
    val_set = SliceDataset('val', root_dir, **common_settings)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=args.num_workers,
                              sampler=RandomSampler(train_set, generator=data_generator),
                              worker_init_fn=seed_worker)
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            num_workers=args.num_workers,
                            shuffle=False,
                            worker_init_fn=seed_worker)

    args.dest.mkdir(parents=True, exist_ok=True)
    return net, optimizer, device, train_loader, val_loader, K


def runTraining(args):
    if args.val_3d and args.val_3d_every < 1:
        raise ValueError('3D validation requires val_3d_every >= 1; use --no_val_3d to disable it')
    print(f">>> Setting up to train on {args.dataset} with {args.net}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)
    loss_fn = make_loss(args, K)

    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))
    log_dice_3d_val: np.ndarray = np.full((args.epochs, K), np.nan, dtype=np.float64)
    log_dice_3d_val_fg: np.ndarray = np.full((args.epochs,), np.nan, dtype=np.float64)
    log_dice_3d_val_patients: dict[str, np.ndarray] = {}

    expected_stems = [pair[0].stem for pair in val_loader.dataset.files]
    history = []
    selected_epoch = None
    updates = 0
    select_by_3d = args.val_3d
    best_score = -1.0
    best_metric_name = "3D foreground DSC" if select_by_3d else "2D foreground DSC"
    if select_by_3d:
        print(">>> Using 3D foreground Dice to select the best epoch.")

    for e in range(args.epochs):
        epoch_record = {'epoch': e}
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            volume_dice = PatientDice(K, expected_stems) if m == 'val' else None
            phase_losses = []
            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    if opt and args.max_updates and updates >= args.max_updates:
                        break
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    assert 0 <= img.min() and img.max() <= 1
                    pred_logits = net(img)
                    pred_probs = F.softmax(pred_logits, dim=1)
                    pred_seg = probs2one_hot(pred_probs)

                    B = img.shape[0]
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)

                    loss = loss_fn(pred_probs, gt, logits=pred_logits)
                    assert torch.isfinite(loss), (e, m, i, loss)
                    log_loss[e, i] = loss.item()
                    phase_losses.append(loss.item())

                    if opt:
                        loss.backward()
                        opt.step()
                        updates += 1

                    if m == 'val':
                        stems = data['stems']
                        volume_dice.update(pred_seg.argmax(1), gt.argmax(1), stems)
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class = probs2class(pred_probs)
                            mult = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        stems,
                                        args.dest / f"iter{e:03d}" / m)

                    j += B
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

            epoch_record[m] = {
                'loss': float(np.mean(phase_losses)),
                'slice_foreground_dice': log_dice[e, :j, 1:].mean().item(),
                'slice_per_organ': log_dice[e, :j].mean(0).tolist(),
                'slices': j,
            }
            if volume_dice is not None:
                epoch_record[m]['patient_volume_dice'] = volume_dice.result()

        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)
        if args.val_3d:
            np.save(args.dest / "dice_3d_val.npy", log_dice_3d_val)
            np.save(args.dest / "dice_3d_val_fg.npy", log_dice_3d_val_fg)
            np.savez(args.dest / "dice_3d_val_patients.npz", **log_dice_3d_val_patients)

        current_2d_dice: float = log_dice_val[e, :, 1:].mean().item()
        current_score = None if select_by_3d else current_2d_dice
        ran_3d_val = False
        temp_3d_volumes: Path | None = None

        final_epoch = e == args.epochs - 1 or (args.max_updates and updates >= args.max_updates)
        if args.val_3d and ((e + 1) % args.val_3d_every == 0 or final_epoch):
            mean_per_class, foreground_mean, per_patient, temp_3d_volumes = run_3d_validation(
                args,
                e,
                K,
                keep_volumes=False,
                cleanup=False,
            )
            log_dice_3d_val[e, :] = mean_per_class
            log_dice_3d_val_fg[e] = foreground_mean
            log_dice_3d_val_patients |= {f"epoch_{e:03d}_{patient}": dice
                                         for patient, dice in per_patient.items()}
            current_score = foreground_mean if select_by_3d else current_2d_dice
            ran_3d_val = True
            print(f">>> 3D validation dice at epoch {e}: {foreground_mean:05.3f} foreground DSC")

        # Unscored epochs are ineligible: sparse 3D validation must never
        # substitute a slice Dice score into the native-volume comparison.
        improved = current_score is not None and np.isfinite(current_score) and current_score > best_score
        if improved:
            message = (f">>> Improved {best_metric_name} at epoch {e}: "
                       f"{best_score:05.3f}->{current_score:05.3f}")
            print(message)
            best_score = current_score
            selected_epoch = e
            (args.dest / "best_epoch.txt").write_text(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", best_folder)

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")

            if args.val_3d and temp_3d_volumes is not None:
                best_3d_root = args.dest / "volumes" / "best_epoch"
                if best_3d_root.exists():
                    rmtree(best_3d_root)
                best_3d_root.mkdir(parents=True, exist_ok=True)
                move(str(temp_3d_volumes), str(best_3d_root / "val"))
                save_3d_validation_metrics(best_3d_root,
                                           e,
                                           mean_per_class,
                                           foreground_mean,
                                           per_patient)
                cleanup_tmp_3d_validation(temp_3d_volumes)
                print(f">>> Promoted best-epoch 3D volumes at epoch {e}: "
                      f"{foreground_mean:05.3f} foreground DSC")
        elif args.val_3d and select_by_3d and temp_3d_volumes is not None:
            cleanup_tmp_3d_validation(temp_3d_volumes)

        if args.val_3d and ran_3d_val:
            np.save(args.dest / "dice_3d_val.npy", log_dice_3d_val)
            np.save(args.dest / "dice_3d_val_fg.npy", log_dice_3d_val_fg)
            np.savez(args.dest / "dice_3d_val_patients.npz", **log_dice_3d_val_patients)
            epoch_record['val']['dice_3d_fg'] = float(log_dice_3d_val_fg[e])

        history.append(epoch_record)
        torch.save(net.state_dict(), args.dest / 'finalweights.pt')
        write_json(args.dest / 'epochs.json', history)
        write_json(args.dest / 'summary.json', {
            'status': 'running',
            'epochs_completed': e + 1,
            'updates': updates,
            'selected_epoch': selected_epoch,
            'selection_metric': 'native_3d_foreground_dice' if select_by_3d else 'slice_foreground_dice',
            'selected_score': best_score if selected_epoch is not None else None,
            'final_epoch': e,
            'selected': history[selected_epoch]['val'] if selected_epoch is not None else None,
            'final': history[-1]['val'],
            'settings': vars(args),
        })
        if args.max_updates and updates >= args.max_updates:
            break

    summary = json.loads((args.dest / 'summary.json').read_text())
    if selected_epoch is None:
        summary['status'] = 'failed'
        write_json(args.dest / 'summary.json', summary)
        raise RuntimeError(f'No finite {best_metric_name} score was available to select a checkpoint')
    summary['status'] = 'complete'
    write_json(args.dest / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save predictions and weights.")
    parser.add_argument('--loss', default='ce', choices=LOSS_NAMES)
    parser.add_argument('--loss_kwargs', default={}, type=json.loads,
                        help="Additional loss kwargs as JSON, such as '{\"boundary_weight\": 0.001}'.")
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd', 'rmsprop', 'adamw'])
    parser.add_argument('--optimizer_kwargs', default={}, type=json.loads,
                        help="Additional optimizer kwargs as JSON, such as '{\"weight_decay\": 0.01}'.")
    parser.add_argument('--net', choices=['shallow', 'enet', 'enet25d'],
                        help="Network to train. Defaults to shallow for TOY2 and ENet for SEGTHOR.")
    parser.add_argument('--slices', type=int, choices=[3, 5],
                        help="Required only for --net enet25d.")
    parser.add_argument('--slice-step', default=1, type=int,
                        help="Spacing between 2.5D context slices.")
    parser.add_argument('--augmentation', default='none', choices=['none', 'mild', 'strong'])
    parser.add_argument('--augmentation-components', nargs='*',
                        choices=['affine', 'gamma', 'noise', 'contrast', 'blur'],
                        help="Optional subset of the selected augmentation policy.")
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--max-updates', default=0, type=int,
                        help="Stop after this many optimizer updates, then validate once.")
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=5, type=int,
                        help="Number of DataLoader workers. Use 0 if multiprocessing/shared memory is unavailable.")
    parser.add_argument('--val_3d', dest='val_3d', action='store_true',
                        help="Run 3D validation by stitching validation predictions. Enabled by default.")
    parser.add_argument('--no_val_3d', dest='val_3d', action='store_false',
                        help="Disable 3D validation and select best epoch by 2D Dice.")
    parser.set_defaults(val_3d=True)
    parser.add_argument('--val_3d_every', default=1, type=int,
                        help="Run native 3D validation every N epochs and at the final epoch; select only among those epochs by 3D Dice.")
    parser.add_argument('--val_3d_gt_root', default=None, type=Path,
                        help="Validation GT root. Defaults to the original source train/ folder.")
    parser.add_argument('--debug', action='store_true',
                        help="Keep only 10 samples per split for quick logic checks.")

    args = parser.parse_args()
    if args.net is None:
        args.net = default_net(args.dataset)

    if args.val_3d and not args.dataset.startswith("SEGTHOR"):
        print(f">>> Disabling 3D validation for {args.dataset}: only SEGTHOR datasets support stitching.")
        args.val_3d = False

    if args.epochs < 1 or args.num_workers < 0 or args.val_3d_every < 0 or args.max_updates < 0:
        parser.error('epochs must be positive; num_workers, val_3d_every and max_updates must be nonnegative')
    if args.val_3d and args.val_3d_every < 1:
        parser.error('--val_3d_every must be positive when 3D validation is enabled')
    if args.batch_size is not None and args.batch_size < 1:
        parser.error('batch-size must be positive')
    if args.slice_step < 1:
        parser.error('slice-step must be positive')
    if args.net == 'enet25d' and args.slices is None:
        parser.error('--net enet25d requires --slices {3,5}')
    if args.net != 'enet25d' and args.slices is not None:
        parser.error('--slices is only valid with --net enet25d')
    if args.net != 'enet25d' and args.slice_step != 1:
        parser.error('--slice-step is only valid with --net enet25d')
    if args.dataset == 'TOY2' and args.net != 'shallow':
        parser.error('TOY2 uses --net shallow')
    if args.dataset == 'TOY2' and args.augmentation != 'none':
        parser.error('TOY2 does not use SEGTHOR augmentation')
    if args.dataset.startswith('SEGTHOR') and args.net == 'shallow':
        parser.error('SEGTHOR uses --net enet or --net enet25d')

    pprint(args)
    runTraining(args)


if __name__ == '__main__':
    main()

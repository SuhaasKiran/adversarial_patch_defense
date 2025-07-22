import os
import torch
from tqdm.notebook import tqdm
from data_utils import download_and_extract, load_dataset, get_label_index
from model_utils import load_pretrained_model, evaluate_model
from patch_utils import patch_attack, evaluate_patch
from defense_utils import initialize_defense, perform_patch_attack_with_defense
from visualization_utils import show_patches, show_table, show_prediction

def main():
    # Configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial10/"
    pretrained_files = [("../data", "TinyImageNet.zip"), ("../saved_models", "patches.zip")]
    class_names = ['toaster', 'goldfish', 'school bus', 'lipstick', 'pineapple']
    patch_sizes = [32, 48, 64]

    # Download and load dataset
    download_and_extract(base_url, pretrained_files)
    dataset, data_loader, label_names = load_dataset()

    # Load pretrained model
    model = load_pretrained_model(device, "../saved_models")
    print(f"Number of testing samples: {len(dataset)}")
    evaluate_model(model, data_loader, device=device)

    # Load or train patches
    patch_dict = {}
    json_results_file = os.path.join("../saved_models", "patch_results.json")
    json_results = {}
    if os.path.isfile(json_results_file):
        with open(json_results_file, "r") as f:
            json_results = json.load(f)

    for name in class_names:
        patch_dict[name] = {}
        for patch_size in patch_sizes:
            c = get_label_index(name, label_names)
            file_name = os.path.join("../saved_models", f"{name}_{patch_size}_patch.pt")
            if not os.path.isfile(file_name):
                patch, val_results = patch_attack(model, c, patch_size, num_epochs=5, dataset=dataset, device=device)
                print(f"Validation results for {name} and {patch_size}: {val_results}")
                torch.save(patch, file_name)
            else:
                patch = torch.load(file_name, map_location=device)
            results = json_results.get(name, {}).get(str(patch_size), evaluate_patch(model, patch, data_loader, c, device))
            patch_dict[name][patch_size] = {"results": results, "patch": patch}

    # Visualize patches and results
    show_patches(patch_dict, class_names, patch_sizes)
    show_table(patch_dict, class_names, patch_sizes, top_1=True)
    show_table(patch_dict, class_names, patch_sizes, top_1=False)

    # Initialize defense
    sac_processor, sac_processor2 = initialize_defense()

    # Evaluate patch attack with defense at different locations
    locations = [
        {"name": "center", "h_offset": 88, "w_offset": 88},
        {"name": "first quadrant", "h_offset": 22, "w_offset": 22},
        {"name": "second quadrant", "h_offset": 22, "w_offset": 154},
        {"name": "third quadrant", "h_offset": 154, "w_offset": 154},
        {"name": "fourth quadrant", "h_offset": 154, "w_offset": 22}
    ]

    total_batches = 47  # For ~1500 samples
    for loc_idx, loc in enumerate(locations):
        count = 0
        total_acc, total_top5, total_patch_acc, total_patch_top5, total_iou = 0.0, 0.0, 0.0, 0.0, 0.0
        for imgs, labels in tqdm(data_loader, desc=f"Validating {loc['name']}..."):
            count += 1
            if count <= loc_idx * 10 or count > total_batches:
                continue
            print(f"Iteration {count} - {loc['name']}")
            imgs, labels = imgs.to(device), labels.to(device)
            acc, top5, patch_acc, patch_top5, iou = perform_patch_attack_with_defense(
                model, patch_dict['goldfish'][48]['patch'], imgs, labels, sac_processor2, device,
                defend=True, h_offset=loc['h_offset'], w_offset=loc['w_offset']
            )
            total_acc += acc
            total_top5 += top5
            total_patch_acc += patch_acc
            total_patch_top5 += patch_top5
            total_iou += iou

        num_batches = 10 if loc_idx < 4 else 7
        print(f"\nResults for {loc['name']} (after {num_batches} batches):")
        print(f"Top-1 accuracy: {total_acc/num_batches:.2f}")
        print(f"Top-5 accuracy: {total_top5/num_batches:.2f}")
        print(f"Top-1 patch accuracy: {total_patch_acc/num_batches:.2f}")
        print(f"Top-5 patch accuracy: {total_patch_top5/num_batches:.2f}")
        print(f"Average IOU: {total_iou/num_batches:.2f}")

    # Visualize predictions for a sample batch
    exmp_batch, label_batch = next(iter(data_loader))
    with torch.no_grad():
        preds = model(exmp_batch.to(device))
    for i in range(1, 20, 5):
        show_prediction(exmp_batch[i], label_batch[i], preds[i], label_names)

if __name__ == "__main__":
    main()
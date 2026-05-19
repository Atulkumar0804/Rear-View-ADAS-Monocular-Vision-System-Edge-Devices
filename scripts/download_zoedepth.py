import torch
import torch.nn as nn
import os
from pathlib import Path

def load_zoedepth_model_fixed(repo="isl-org/ZoeDepth", model_name="ZoeD_K"):
    """
    Loads ZoeDepth model with a fix for recent timm versions.
    """
    print(f"Loading {model_name} from {repo}...")
    # Load model structure without weights
    model = torch.hub.load(repo, model_name, pretrained=False)
    
    # Define URL for pretrained weights
    if model_name == "ZoeD_K":
        url = "https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_K.pt"
    elif model_name == "ZoeD_NK":
        url = "https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt"
    elif model_name == "ZoeD_N":
        url = "https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_N.pt"
    else:
        return torch.hub.load(repo, model_name, pretrained=True)

    # Load state dict
    print(f"Downloading/Loading weights from {url}...")
    pretrained_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
    
    # Load weights with strict=False to ignore mismatching keys
    print("Loading state dict with strict=False...")
    model.load_state_dict(pretrained_dict['model'], strict=False)
    
    # Monkey patch the blocks to fix the timm issue
    print("Applying timm compatibility fix...")
    try:
        for b in model.core.core.pretrained.model.blocks:
            b.drop_path = nn.Identity()
        print("✅ Applied timm compatibility fix.")
    except AttributeError:
        print("⚠️ Could not apply timm compatibility fix.")

    return model

def download_zoedepth():
    print("⬇️  Downloading ZoeDepth (ZoeD_K for KITTI/Outdoor)...")
    
    # Define cache directory to keep models organized
    cache_dir = Path("models/zoedepth")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set torch hub directory
    torch.hub.set_dir(str(cache_dir))
    
    try:
        # Load the model to trigger download using the fixed loader
        model = load_zoedepth_model_fixed("isl-org/ZoeDepth", "ZoeD_K")
        print("✅ ZoeDepth (ZoeD_K) downloaded successfully!")
        print(f"📂 Model cached in: {cache_dir}")
    except Exception as e:
        print(f"❌ Failed to download ZoeDepth: {e}")
        print("   Please check your internet connection and try again.")

if __name__ == "__main__":
    download_zoedepth()

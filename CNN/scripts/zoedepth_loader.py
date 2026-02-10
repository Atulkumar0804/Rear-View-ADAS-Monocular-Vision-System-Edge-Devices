import torch
import torch.nn as nn
from pathlib import Path

def load_zoedepth_model(model_name="ZoeD_K", device="cuda"):
    """
    Loads ZoeDepth model with a fix for recent timm versions.
    """
    print(f"📦 Loading ZoeDepth ({model_name})...")
    
    # Define cache directory
    cache_dir = Path("models/zoedepth")
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache_dir))
    
    try:
        # 1. Load model structure without weights first
        # We use the torch.hub interface but disable pretraining to avoid the error
        repo = "isl-org/ZoeDepth"
        model = torch.hub.load(repo, model_name, pretrained=False)
        
        # 2. Manually load weights with strict=False
        finetuned_dir = Path("models/zoedepth_finetuned")
        finetuned_weights = None
        if finetuned_dir.exists():
             checkpoints = sorted(finetuned_dir.glob("zoedepth_epoch_*.pt"), 
                                key=lambda x: int(x.stem.split('_')[-1]))
             if checkpoints:
                 finetuned_weights = checkpoints[-1]
        
        if finetuned_weights and model_name == "ZoeD_NK":
             print(f"🔥 Loading fine-tuned weights from {finetuned_weights}...")
             checkpoint = torch.load(finetuned_weights, map_location='cpu')
        elif model_name == "ZoeD_K":
            url = "https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_K.pt"
            print(f"   Downloading/Loading weights from {url}...")
            checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu', progress=True)
        elif model_name == "ZoeD_NK":
            url = "https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt"
            print(f"   Downloading/Loading weights from {url}...")
            checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu', progress=True)
        else:
            # Fallback for other variants
            model = torch.hub.load(repo, model_name, pretrained=True)
            return model.to(device)

        
        # The checkpoint usually has a 'model' key
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # Load with strict=False to ignore the 'relative_position_index' mismatch
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"   Weights loaded (Missing keys expected for timm compat): {len(msg.missing_keys)}")
        
        # 3. Apply fix for newer timm versions if needed
        # Newer timm versions might have issues with DropPath in this specific architecture
        # This is a common workaround for ZoeDepth + timm > 0.9
        try:
            # Navigate to the backbone blocks
            # Structure: model -> core -> core -> pretrained -> model -> blocks
            if hasattr(model, 'core') and hasattr(model.core, 'core'):
                blocks = model.core.core.pretrained.model.blocks
                print(f"   Patching {len(blocks)} blocks for timm compatibility...")
                for i, block in enumerate(blocks):
                    if not hasattr(block, 'drop_path'):
                        # Inject Identity if missing
                        block.drop_path = nn.Identity()
                    elif isinstance(block.drop_path, nn.Module):
                        # If it is a module (like DropPath), replace with Identity to be safe/deterministic
                        # or leave it if it works. But since we had issues, let's force Identity.
                        block.drop_path = nn.Identity()
                    else:
                        # If it's something else (like a function in older timm?), make it Identity module
                        block.drop_path = nn.Identity()
                        
        except Exception as e:
            print(f"   ⚠️ Note: Could not apply DropPath fix (might not be needed): {e}")

        model.to(device)
        model.eval()
        print(f"✅ ZoeDepth ({model_name}) loaded successfully!")
        return model

    except Exception as e:
        print(f"❌ Failed to load ZoeDepth: {e}")
        return None

if __name__ == "__main__":
    # Test the loader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_zoedepth_model("ZoeD_K", device)
    
    if model:
        print("Successfully loaded ZoeD_K!")
        # Optional: Test inference with a dummy input
        print("Testing inference with dummy input...")
        try:
            dummy_input = torch.rand(1, 3, 384, 512).to(device)
            with torch.no_grad():
                out = model(dummy_input)
            print(f"Inference successful! Output shape: {out['metric_depth'].shape}")
        except Exception as e:
            print(f"Inference failed: {e}")

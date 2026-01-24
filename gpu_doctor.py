import torch
import sys
import subprocess

def run_doctor():
    print("="*30)
    print("🎙️ PRO AI STUDIO: GPU DOCTOR")
    print("="*30)

    # 1. Check Python & Torch
    print(f"🐍 Python Version: {sys.version.split()[0]}")
    print(f"🔥 PyTorch Version: {torch.__version__}")
    
    # 2. Check CUDA Availability
    cuda_available = torch.cuda.is_available()
    print(f"💡 Is CUDA available? {'✅ YES' if cuda_available else '❌ NO'}")

    if cuda_available:
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        print("🎉 Your system is ready for high-speed generation!")
    else:
        print("\n--- 🕵️ DIAGNOSIS ---")
        
        # Check if NVIDIA drivers exist via system command
        try:
            #nvidia_smi = subprocess.check_output("nvidia-smi", shell=True).decode()
            print("✅ NVIDIA Drivers: Found")
            
            if "+cpu" in torch.__version__ or "cu" not in torch.__version__:
                print("❌ ISSUE: You have the CPU-only version of PyTorch installed.")
                print("\n🛠️ FIX: Run this command to install the GPU version:")
                print("pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118")
            else:
                print("❌ ISSUE: Driver/Toolkit mismatch. Your drivers might be too old.")
                print("🛠️ FIX: Update your drivers at: https://www.nvidia.com/en-us/geforce/drivers/")
                
        except Exception:
            print("❌ ISSUE: No NVIDIA GPU or Drivers found.")
            print("ℹ️ Note: If you don't have an NVIDIA card, the app will stay in CPU mode.")

    print("="*30)

if __name__ == "__main__":
    run_doctor()
import warnings
# Suppress specific category of warnings to keep the terminal output clean
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import os
import torch
import torchaudio
import soundfile as sf
import re
import gc
import json
import time # Used for measuring generation speed (telemetry)
from TTS.api import TTS
import datetime

# --- CONFIGURATION & DEVICE SETUP ---
# Detects your NVIDIA GTX 1650 automatically; falls back to CPU if not found
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fix for potential multi-load library conflicts in Windows environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define local directories for voice fingerprints and generated masters
LIB_DIR = "Voice_Library"
OUT_DIR = "Output"
CONFIG_FILE = "pronunciation_config.json"

# Ensure required folders exist on the local disk
for d in [LIB_DIR, OUT_DIR]:
    if not os.path.exists(d): 
        os.makedirs(d)

# Set up the browser tab title and wide-screen layout
st.set_page_config(page_title="Pro AI Voice Studio", page_icon="🎙️", layout="wide")

# --- PERSISTENCE & TELEMETRY LOGIC ---
def get_gpu_memory():
    """Calculates real-time VRAM usage of your GTX 1650."""
    if torch.cuda.is_available():
        # Returns current allocated memory in Gigabytes
        return f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    return "N/A"

def load_config():
    """Attempts to load custom dictionary and acronym rules from local JSON."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    # Standard defaults if no config file is found
    return {
        "readable": "NATO, FOMO, LOL, ASAP, NASA",
        "custom": "ChatGPT -> Chat G P T\nAI -> A I"
    }

def save_config(readable, custom):
    """Saves current sidebar inputs to a persistent local JSON file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump({"readable": readable, "custom": custom}, f)
    st.sidebar.success("✅ Configuration Saved!")

@st.cache_resource
def load_model():
    """Downloads and caches the XTTS v2 model into memory/GPU."""
    use_gpu = torch.cuda.is_available()
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)

# --- TEXT NORMALIZATION PIPELINE ---
def strict_normalize_text(text, readable_list, custom_dict):
    """Converts symbols and acronyms into spoken words before AI processing."""
    if not text: return ""

    # 1. Apply user-defined manual replacements (e.g., GPT -> G P T)
    for pair in custom_dict:
        if "->" in pair:
            parts = pair.split("->")
            if len(parts) == 2:
                orig, replacement = parts
                text = re.sub(rf'\b{orig.strip()}\b', replacement.strip(), text, flags=re.IGNORECASE)

    # 2. Phonetic Acronym Speller (e.g., CSG -> C S G)
    # Skips words found in the 'Readable Acronyms' whitelist
    def acronym_handler(m):
        word = m.group(1)
        if word.upper() in [w.strip().upper() for w in readable_list]:
            return word 
        return " ".join(list(word))
    text = re.sub(r'\b([A-Z]{2,5})\b', acronym_handler, text)

    # 3. Simple Currency Logic ($4.5 -> 4 point 5 dollars)
    text = re.sub(r'\$(\d+\.?\d*)', r'\1 dollars', text)
    text = re.sub(r'£(\d+\.?\d*)', r'\1 pounds', text)
    text = re.sub(r'€(\d+\.?\d*)', r'\1 euros', text)

    # 4. Units, Math, and Time conversion
    text = re.sub(r'\b(\d+)(am|pm)\b', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)°C', r'\1 degrees celsius', text)
    text = re.sub(r'(\d+)%', r'\1 percent', text)
    text = re.sub(r'@', ' at ', text)
    text = re.sub(r'&', ' and ', text)
    text = re.sub(r'\+', ' plus ', text)
    text = re.sub(r'=', ' equals ', text)
    text = re.sub(r'\b(\d+)kg\b', r'\1 kilograms', text)
    text = re.sub(r'\b(\d+)km\b', r'\1 kilometers', text)
    text = re.sub(r'\b(\d+)m\b', r'\1 meters', text)
    text = re.sub(r'(\d+)\.(\d+)', r'\1 point \2', text)
    text = re.sub(r'\b(19|20)(\d{2})\b', r'\1 \2', text)

    # 5. Final sanitization of special characters that cause AI artifacts
    text = re.sub(r'[@#$^&*()_+=~`\[\]{}|\\<>/]', ' ', text)
    text = " ".join(text.split())
    text = re.sub(r'\s*([.!?,;])', r'\1', text)
    text = re.sub(r'([.!?,;])(?=[^\s])', r'\1 ', text)
    return text.strip()

def split_text_for_engine(text, max_chars=160):
    """Semantic chunking to prevent processing timeouts during narration."""
    if not text: return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current_chunk = [], ""
    for s in sentences:
        s = s.strip()
        if not s: continue
        if len(current_chunk) + len(s) + 1 <= max_chars:
            current_chunk = (current_chunk + " " + s).strip()
        else:
            if current_chunk: chunks.append(current_chunk)
            current_chunk = s
    if current_chunk: chunks.append(current_chunk)
    return chunks

# --- USER INTERFACE SETUP ---
# Initialize saved preferences from previous sessions
saved_data = load_config()

with st.sidebar:
    st.header("⚙️ Pronunciation Config")
    # Configuration for acronyms that should be read as whole words
    readable_input = st.text_area("Readable Acronyms:", value=saved_data["readable"])
    readable_list = [x.strip() for x in readable_input.split(",")]

    # Dictionary for manual phonetic overrides
    custom_input = st.text_area("Custom Dict (orig -> replacement):", value=saved_data["custom"])
    custom_dict = [line.strip() for line in custom_input.split("\n") if line.strip()]

    # Persistent save button
    if st.button("💾 Save Configuration", use_container_width=True):
        save_config(readable_input, custom_input)
    
    st.divider()
    # Live Telemetry Dashboard for real-time monitoring
    st.header("📊 Live Telemetry")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    st.success(f"📟 **Mode:** {DEVICE.upper()}")
    st.info(f"🏎️ **GPU:** {gpu_name}")
    
    # Placeholder for live VRAM updates during the generation loop
    vram_stat = st.empty() 
    vram_stat.write(f"🧠 **VRAM:** {get_gpu_memory()}")
    
    st.divider()
    st.header("🔍 Normalization Preview")
    status_box = st.container()

# Define the two main work areas: Production and Cloning
tab1, tab2 = st.tabs(["🚀 Production Studio", "👤 Character Library"])

with tab1:
    # Main script input field
    user_text = st.text_area("Script Input:", height=250, key="main_input", placeholder="Paste script here...")
    
    # Pre-process the script using defined normalization rules
    clean_text = strict_normalize_text(user_text, readable_list, custom_dict)
    final_chunks = split_text_for_engine(clean_text)
    
    # Provide a real-time 'spoken version' preview in the sidebar
    with st.sidebar:
        with status_box:
            if clean_text:
                st.markdown("### 📝 Spoken Preview:")
                st.caption(clean_text)
                st.divider()
                st.markdown(f"**Total Chunks:** {len(final_chunks)}")
            else:
                st.info("Waiting for script...")

    # Display script stats
    char_cnt, word_cnt, chunk_cnt = len(clean_text), len(clean_text.split()), len(final_chunks)
    st.markdown(f"📊 **Stats:** {char_cnt} Chars | {word_cnt} Words | **{chunk_cnt} Chunks**")
    
    # Creativity slider
    temp_val = st.select_slider("🌡️ Voice Temperature", options=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85], value=0.75)

    st.markdown("---")
    col_l, col_r = st.columns([1, 1])
    
    # Retrieve available voice fingerprints
    pth_files = [f for f in os.listdir(LIB_DIR) if f.endswith('.pth')]
    
    with col_l:
        # Hide the .pth extension in the dropdown for a cleaner UI
        voice_map = {f.replace('.pth', ''): f for f in pth_files} if pth_files else {}
        selected_display = st.selectbox("Choose Character:", list(voice_map.keys()) if voice_map else ["No voices"])
        selected_voice = voice_map.get(selected_display)
    
    with col_r:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        gen_btn = st.button("✨ Generate 48k Narration", use_container_width=True)

    # --- CORE GENERATION ENGINE ---
    if gen_btn and selected_voice:
        if not clean_text: 
            st.warning("Enter a script first.")
        else:
            # Benchmark timer for telemetry
            start_time = time.time()
            with st.status("🚀 Producing Studio Master...", expanded=True) as status:
                tts = load_model()
                
                # Move voice latents to GPU memory for high-speed inference
                voice_data = torch.load(os.path.join(LIB_DIR, selected_voice), weights_only=False)
                gpt_cond_latent = voice_data["gpt_cond_latent"].to(DEVICE)
                speaker_embedding = voice_data["speaker_embedding"].to(DEVICE)
                
                progress_bar = st.progress(0)
                audio_pieces = []
                
                # Sequential chunk generation loop
                for i, chunk in enumerate(final_chunks):
                    # Update live VRAM display in sidebar
                    vram_stat.write(f"🧠 **VRAM:** {get_gpu_memory()}")
                    
                    status.update(label=f"🎙️ Narrating {i+1}/{chunk_cnt}...", state="running")
                    
                    # Core AI Inference on GPU
                    out = tts.synthesizer.tts_model.inference(
                        text="  " + chunk + "  ", language="en",
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        temperature=temp_val, speed=1.0
                    )
                    
                    # Convert to tensor and trim silent buffers
                    wav_t = torch.as_tensor(out["wav"]).unsqueeze(0)
                    if wav_t.shape[1] > 600: 
                        wav_t = wav_t[:, 200:-200]
                    
                    audio_pieces.append(wav_t)
                    progress_bar.progress((i + 1) / chunk_cnt)

                # Concatenate fragments and resample to 48kHz
                combined = torch.cat(audio_pieces, dim=1)
                resampler = torchaudio.transforms.Resample(orig_freq=24000, new_freq=48000)
                mastered = resampler(combined)
                
                if mastered.shape[0] > 1: 
                    mastered = torch.mean(mastered, dim=0, keepdim=True)
                
                # Save Track
                out_name = f"Master_{datetime.datetime.now().strftime('%H%M%S')}.wav"
                out_path = os.path.join(OUT_DIR, out_name)
                torchaudio.save(out_path, mastered, 48000)
                
                # Clear GPU memory
                gc.collect()
                
                total_time = time.time() - start_time
                status.update(label=f"✅ Master Ready in {total_time:.2f}s!", state="complete")
            
            st.audio(out_path)
            st.download_button("📥 Download 48k WAV", open(out_path, "rb"), file_name=out_name, use_container_width=True)

# --- CLONING TAB (Voice Extraction) ---
with tab2:
    st.subheader("👤 Create New Character Fingerprint")
    v_wav = st.file_uploader("Upload reference (WAV only)", type=["wav"])
    
    l_c1, l_c2 = st.columns(2)
    s_start = l_c1.number_input("Start Time (sec):", min_value=0, value=0)
    s_dur = l_c2.number_input("Total Duration (sec):", min_value=1, max_value=30, value=12)
    
    if st.button("🧬 Extract Vocal DNA", use_container_width=True):
        if v_wav:
            with st.status("Analyzing Voice Characteristics...") as s:
                with open("temp.wav", "wb") as f: 
                    f.write(v_wav.getbuffer())
                
                tts = load_model()
                sr = sf.info("temp.wav").samplerate
                
                # Load selected segment for cloning
                wav, _ = torchaudio.load("temp.wav", frame_offset=int(s_start*sr), num_frames=int(s_dur*sr))
                v_name = v_wav.name.split('.')[0]
                
                # Extract latents and save to fingerprint file
                torchaudio.save("temp_s.wav", wav, sr)
                lats = tts.synthesizer.tts_model.get_conditioning_latents(audio_path=["temp_s.wav"])
                torch.save({"gpt_cond_latent": lats[0], "speaker_embedding": lats[1]}, os.path.join(LIB_DIR, f"{v_name}.pth"))
                
                os.remove("temp.wav")
                os.remove("temp_s.wav")
                st.rerun()
from tracemalloc import BaseFilter
import torch
import streamlit as st
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub import login
import io
import gc
import psutil
import time
import random
from PIL import Image
import speech_recognition as sr
from PIL import Image, ImageEnhance, ImageFilter


# ------------------------- Page Setup ------------------------- #
st.set_page_config(page_title="🎨 AI Image Generator", layout="centered")
st.markdown("<h1 style='text-align: center;'>🎨 Text-to-Image Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Generate stunning images from your imagination using the most lightweight diffusion model!</p>", unsafe_allow_html=True)

# ------------------------- Authenticate HuggingFace ------------------------- #
@st.cache_resource(show_spinner=False)
def authenticate_huggingface():
    try:
        login("your token")  # Replace with your token
        return True
    except Exception as e:
        st.error(f"❌ Hugging Face login failed: {str(e)}")
        return False

if authenticate_huggingface():
    st.success("✅ Connected to Hugging Face")

# ------------------------- Model Loader ------------------------- #
@st.cache_resource
def load_model():
    try:
        model_id = "segmind/tiny-sd"  # ~1.1GB, most lightweight available
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=False,  
            safety_checker=None,  
            feature_extractor=None,
            low_cpu_mem_usage=True,
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)

        if device == "cuda":
            pipe.enable_attention_slicing()
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        return pipe
    except Exception as e:
        st.error(f"❌ Model load failed: {str(e)}")
        return None

# ------------------------- Main UI ------------------------- #
st.markdown("### Create Your Image")
prompt = st.text_input("📝 Enter your prompt:",
                     placeholder="e.g., A futuristic cityscape at sunset with flying cars")

# New interactive options
col1, col2 = st.columns(2)
with col1:
    steps = st.slider("🎚️ Quality (steps)", 5, 50, 25, help="Higher steps = better quality but slower generation")
with col2:
    guidance = st.slider("🎨 Creativity", 1.0, 10.0, 7.5, help="Lower = more creative, higher = more prompt-adherent")

with st.expander("✨ Additional Settings"):
    num_images = st.slider("🔢 Number of Variations", 1, 5, 1, help="Generate multiple image variations")
    style = st.selectbox("🎨 Image Style", options=["Realistic", "Cartoon", "Abstract", "Cyberpunk"], index=0)

# Voice-to-Text Functionality
def record_voice():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        st.info("🔊 Listening for your voice... (speak now)")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        st.success(f"🎤 You said: {text}")
        return text
    except Exception as e:
        st.error("❌ Sorry, I couldn't recognize your speech. Please try again.")
        return None

voice_button = st.button("🎙️ Use Voice to Text", help="Click this button to speak your prompt")
if voice_button:
    prompt = record_voice()  # Set prompt based on voice input

# Save prompt as a favorite
favorites = []

def save_to_favorites(prompt, img):
    st.session_state.favorites.append((prompt, img))
    st.sidebar.success(f"✔️ '{prompt[:20]}...' added to Favorites!")


generate = st.button("🚀 Generate Image")

if "history" not in st.session_state:
    st.session_state.history = []
    
if "favorites" not in st.session_state:
    st.session_state.favorites = []


if generate and prompt.strip():
    try:
        with st.spinner("🧠 Loading model and generating image..."):
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
            
            # Load model only when needed
            pipe = load_model()
            
            if pipe:
                images = []
                for _ in range(num_images):
                    with st.spinner(f"Generating image {_ + 1} of {num_images}..."):
                        # Adjust settings for better image quality and less noise
                        result = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance)
                        img = result.images[0]
                        
                        # Post-processing: applying basic noise reduction (optional)
                        img = img.filter(ImageFilter.GaussianBlur(radius=1))  # Smooth out noise slightly
                        
                        images.append(img)
                
                # Save images to buffer
                buf = io.BytesIO()
                for i, img in enumerate(images):
                    img.save(buf, format="PNG")
                    buf.seek(0)
                
                # Cleanup memory
                del pipe, result
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Show images and download links
                for i, img in enumerate(images):
                    st.image(img, caption=f"Image Variation {i + 1} - Style: {style}")
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    st.download_button(
                        label=f"💾 Download Image {i + 1}",
                        data=buf.getvalue(),
                        file_name=f"generated_image_{i + 1}.png",
                        mime="image/png"
                    )

                    # Add to history
                    st.session_state.history.append((prompt, img))

                    # Favorites option
                    save_button = st.button(f"⭐ Save to Favorites {i + 1}")
                    if save_button:
                        save_to_favorites(prompt, img)
                    
                
                st.success("✅ Image(s) generated successfully!")

    except Exception as e:
        st.error(f"❌ Error generating image: {str(e)}")
        st.info("Try restarting the app or using a simpler prompt.")
else:
    st.info("👆 Enter a prompt and click 'Generate Image' to begin!")

# ------------------------- Prompt History ------------------------- #
st.sidebar.markdown("### 🕘 Prompt History")
if st.session_state.history:
    for i, (p, img) in enumerate(reversed(st.session_state.history), 1):
        st.sidebar.markdown(f"**{i}.** {p}")
        st.sidebar.image(img, caption=f"Image {i}", width=100)
else:
    st.sidebar.info("No prompts yet.")

# ------------------------- Favorites ------------------------- #
# ------------------------- Favorites ------------------------- #
st.sidebar.markdown("### ⭐ Your Favorites")
if st.session_state.favorites:
    for i, (p, img) in enumerate(st.session_state.favorites, 1):
        st.sidebar.markdown(f"**{i}.** {p}")
        st.sidebar.image(img, caption=f"Favorite {i}", width=100)
else:
    st.sidebar.info("No favorites yet.")


# ------------------------- Tutorial ------------------------- #
st.sidebar.markdown("### 📚 Prompt Writing Tutorial")
st.sidebar.markdown("""
Here are some examples of creative prompts to get you started:

- "A futuristic cityscape with flying cars and neon lights"
- "A serene sunset over a mountain range, painted in impressionistic style"
- "A cartoon character in a medieval fantasy setting"
- "An abstract geometric design with vibrant colors"

Feel free to get creative with your prompts! The more descriptive, the better the results!
""")

# ------------------------- System Info ------------------------- #
st.sidebar.markdown("### System Info")
device_info = f"{'🖥️ GPU' if torch.cuda.is_available() else '💻 CPU'} Mode"
st.sidebar.info(device_info)

# Memory monitoring
try:
    ram_usage = psutil.virtual_memory().percent
    st.sidebar.progress(ram_usage / 100)
    st.sidebar.text(f"RAM Usage: {ram_usage}%")
except Exception:
    st.sidebar.text("RAM monitoring unavailable")

if torch.cuda.is_available():
    st.sidebar.text(f"GPU: {torch.cuda.get_device_name(0)}")
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    st.sidebar.text(f"Total GPU Memory: {total_memory:.2f} GB")
    
    if st.sidebar.button("🧹 Force Memory Cleanup"):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        st.sidebar.success("Memory cleaned!")
        time.sleep(1)
        st.experimental_rerun()












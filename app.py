import argparse

import gradio as gr
from PIL import Image
import os
import torch
import numpy as np
import cv2
import yaml
from huggingface_hub import hf_hub_download

# Local code
from models import instructir
from text.models import LanguageModel, LMHead


# =========================================================
# CONFIGURATION
# =========================================================

def dict2namespace(config):
    namespace = argparse.Namespace()

    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value

        setattr(namespace, key, new_value)

    return namespace


# =========================================================
# DOWNLOAD MODEL FILES
# =========================================================

hf_hub_download(
    repo_id="marcosv/InstructIR",
    filename="im_instructir-7d.pt",
    local_dir="./"
)

hf_hub_download(
    repo_id="marcosv/InstructIR",
    filename="lm_instructir-7d.pt",
    local_dir="./"
)


CONFIG = "configs/eval5d.yml"
LM_MODEL = "lm_instructir-7d.pt"
MODEL_NAME = "im_instructir-7d.pt"


# =========================================================
# LOAD CONFIG
# =========================================================

with open(CONFIG, "r") as f:
    config = yaml.safe_load(f)

cfg = dict2namespace(config)


# =========================================================
# DEVICE
# =========================================================

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# =========================================================
# LOAD IMAGE RESTORATION MODEL
# =========================================================

model = instructir.create_model(
    input_channels=cfg.model.in_ch,
    width=cfg.model.width,
    enc_blks=cfg.model.enc_blks,
    middle_blk_num=cfg.model.middle_blk_num,
    dec_blks=cfg.model.dec_blks,
    txtdim=cfg.model.textdim
)

model = model.to(device)

print("IMAGE MODEL CKPT:", MODEL_NAME)

model.load_state_dict(
    torch.load(MODEL_NAME, map_location="cpu"),
    strict=True
)


# =========================================================
# LOAD LANGUAGE MODEL
# =========================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LMODEL = cfg.llm.model

language_model = LanguageModel(
    model=LMODEL
)

lm_head = LMHead(
    embedding_dim=cfg.llm.model_dim,
    hidden_dim=cfg.llm.embd_dim,
    num_classes=cfg.llm.nclasses
)

lm_head = lm_head.to(device)

print("LMHEAD MODEL CKPT:", LM_MODEL)

lm_head.load_state_dict(
    torch.load(LM_MODEL, map_location="cpu"),
    strict=True
)


# =========================================================
# IMAGE PROCESSING FUNCTION
# =========================================================

def load_img(filename, norm=True):

    img = np.array(
        Image.open(filename).convert("RGB")
    )

    if norm:
        img = img / 255.
        img = img.astype(np.float32)

    return img


def process_img(image, prompt):

    if image is None:
        return None

    if prompt is None or prompt.strip() == "":
        prompt = "Enhance the image."

    img = np.array(image)

    img = img / 255.
    img = img.astype(np.float32)

    y = torch.tensor(img).permute(
        2, 0, 1
    ).unsqueeze(0).to(device)

    lm_embd = language_model(prompt)
    lm_embd = lm_embd.to(device)

    with torch.no_grad():

        text_embd, deg_pred = lm_head(lm_embd)

        x_hat = model(
            y,
            text_embd
        )

    restored_img = (
        x_hat.squeeze()
        .permute(1, 2, 0)
        .clamp_(0, 1)
        .cpu()
        .detach()
        .numpy()
    )

    restored_img = np.clip(
        restored_img,
        0.,
        1.
    )

    restored_img = (
        restored_img * 255.0
    ).round().astype(np.uint8)

    return Image.fromarray(restored_img)


# =========================================================
# OBJECT REMOVAL FUNCTION
# =========================================================

def remove_object(image, mask):

    if image is None or mask is None:
        return image

    image_np = np.array(image).astype(np.uint8)

    mask_np = np.array(mask).astype(np.uint8)

    # Handle RGBA mask
    if len(mask_np.shape) == 3:

        if mask_np.shape[2] == 4:
            # Use alpha channel for painted area
            alpha = mask_np[:, :, 3]

            if np.max(alpha) > 0:
                mask_np = alpha
            else:
                mask_np = cv2.cvtColor(
                    mask_np[:, :, :3],
                    cv2.COLOR_RGB2GRAY
                )

        else:
            mask_np = cv2.cvtColor(
                mask_np,
                cv2.COLOR_RGB2GRAY
            )

    # Create binary mask
    mask_np = np.where(
        mask_np > 10,
        255,
        0
    ).astype(np.uint8)

    # Inpainting
    result = cv2.inpaint(
        image_np,
        mask_np,
        5,
        cv2.INPAINT_TELEA
    )

    return Image.fromarray(result)


# =========================================================
# OBJECT REMOVAL PROCESSING
# =========================================================

def remove_selected_object(editor_data):

    if editor_data is None:
        return None

    try:

        # ImageEditor returns a dictionary
        if isinstance(editor_data, dict):

            background = editor_data.get("background")
            layers = editor_data.get("layers", [])

            if background is None:
                return None

            if not layers:
                return background

            mask = layers[0]

        else:
            return None

        if mask is None:
            return background

        # Convert mask to numpy
        mask_np = np.array(mask).astype(np.uint8)

        # Handle RGBA layer
        if len(mask_np.shape) == 3:

            if mask_np.shape[2] == 4:

                alpha = mask_np[:, :, 3]

                if np.max(alpha) > 0:
                    mask_np = alpha
                else:
                    mask_np = cv2.cvtColor(
                        mask_np[:, :, :3],
                        cv2.COLOR_RGB2GRAY
                    )

            else:

                mask_np = cv2.cvtColor(
                    mask_np,
                    cv2.COLOR_RGB2GRAY
                )

        # Make mask binary
        mask_np = np.where(
            mask_np > 10,
            255,
            0
        ).astype(np.uint8)

        # Perform object removal
        result = remove_object(
            background,
            mask_np
        )

        return result

    except Exception as e:

        print("Object removal error:", e)

        return None


# =========================================================
# CUSTOM CSS
# =========================================================

custom_css = """

body {
    background: #f5f5f5;
}

.gradio-container {
    max-width: 1050px !important;
    margin: auto !important;
    padding-top: 25px !important;
}

.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 8px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    color: #666666;
    margin-bottom: 35px;
}

.card {
    background: white;
    border-radius: 18px;
    padding: 32px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.section-title {
    text-align: center;
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 10px;
}

.section-text {
    text-align: center;
    font-size: 16px;
    color: #666666;
    margin-bottom: 25px;
}

.step-text {
    text-align: center;
    font-size: 14px;
    color: #888888;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

button {
    border-radius: 10px !important;
    min-height: 42px !important;
}

.image-frame {
    border-radius: 12px;
}

.exit-message {
    text-align: center;
    padding: 25px;
    font-size: 18px;
}

"""


# =========================================================
# GRADIO APPLICATION
# =========================================================

with gr.Blocks(
    css=custom_css,
    title="Smart Image Enhancement System"
) as demo:


    # =====================================================
    # WELCOME PAGE
    # =====================================================

    with gr.Column(
        visible=True
    ) as welcome_page:

        gr.Markdown(
            """
            <div class="main-title">
                Smart Image Enhancement
            </div>

            <div class="subtitle">
                Instruction-Guided Image Restoration using Deep Learning
            </div>
            """
        )

        with gr.Column(
            elem_classes="card"
        ):

            gr.Markdown(
                """
                <div class="section-title">
                    Hi, welcome to the image enhancement process.
                </div>

                <div class="section-text">
                    Restore and enhance your degraded images using
                    intelligent deep learning-based image restoration.
                </div>
                """
            )

            with gr.Row():

                start_btn = gr.Button(
                    "START",
                    variant="primary"
                )

                cancel_btn = gr.Button(
                    "CANCEL"
                )

            exit_message = gr.Markdown(
                "",
                elem_classes="exit-message"
            )


    # =====================================================
    # UPLOAD PAGE
    # =====================================================

    with gr.Column(
        visible=False
    ) as upload_page:

        gr.Markdown(
            """
            <div class="step-text">
                STEP 1 OF 3
            </div>

            <div class="section-title">
                Upload Your Image
            </div>

            <div class="section-text">
                Select the image you want to enhance.
            </div>
            """
        )

        with gr.Column(
            elem_classes="card"
        ):

            image_input = gr.Image(
                type="pil",
                label="Input Image"
            )

            with gr.Row():

                upload_back_btn = gr.Button(
                    "BACK"
                )

                upload_next_btn = gr.Button(
                    "NEXT",
                    variant="primary"
                )


    # =====================================================
    # INSTRUCTION PAGE
    # =====================================================

    with gr.Column(
        visible=False
    ) as instruction_page:

        gr.Markdown(
            """
            <div class="step-text">
                STEP 2 OF 3
            </div>

            <div class="section-title">
                Tell Us What You Want to Fix
            </div>

            <div class="section-text">
                Describe the enhancement you want to perform.
            </div>
            """
        )

        with gr.Column(
            elem_classes="card"
        ):

            image_preview = gr.Image(
                type="pil",
                label="Selected Image",
                interactive=False
            )

            prompt_input = gr.Textbox(
                label="Enhancement Instruction",
                placeholder="Example: Remove the raindrops from the image",
                lines=2
            )

            gr.Markdown(
                "### Restoration Suggestions"
            )

            with gr.Row():

                noise_btn = gr.Button(
                    "Remove Noise"
                )

                blur_btn = gr.Button(
                    "Remove Blur"
                )

                rain_btn = gr.Button(
                    "Remove Raindrops"
                )

            with gr.Row():

                haze_btn = gr.Button(
                    "Remove Haze"
                )

                light_btn = gr.Button(
                    "Improve Lighting"
                )


            # =================================================
            # OBJECT REMOVAL
            # =================================================

            gr.Markdown(
                "### Selective Object Removal"
            )

            gr.Markdown(
                "Paint over the object you want to remove."
            )

            object_editor = gr.ImageEditor(
                label="Select Object",
                type="pil"
            )

            remove_object_btn = gr.Button(
                "REMOVE SELECTED OBJECT",
                variant="primary"
            )


            with gr.Row():

                instruction_back_btn = gr.Button(
                    "BACK"
                )

                enhance_btn = gr.Button(
                    "ENHANCE IMAGE",
                    variant="primary"
                )


    # =====================================================
    # PROCESSING PAGE
    # =====================================================

    with gr.Column(
        visible=False
    ) as processing_page:

        gr.Markdown(
            """
            <div class="section-title">
                Enhancing Your Image
            </div>

            <div class="section-text">
                Your image is being restored using deep learning.
            </div>
            """
        )

        with gr.Column(
            elem_classes="card"
        ):

            processing_text = gr.Markdown(
                """
                <div style="text-align:center; padding:30px; font-size:18px;">
                    Processing image...<br><br>
                    Please wait while the image is being enhanced.
                </div>
                """
            )


    # =====================================================
    # RESULT PAGE
    # =====================================================

    with gr.Column(
        visible=False
    ) as result_page:

        gr.Markdown(
            """
            <div class="section-title">
                Enhancement Complete
            </div>

            <div class="section-text">
                Your enhanced image is ready.
            </div>
            """
        )

        with gr.Row():

            with gr.Column(
                elem_classes="card"
            ):

                gr.Markdown(
                    "### Original Image"
                )

                original_result = gr.Image(
                    type="pil",
                    label="Original",
                    interactive=False
                )

            with gr.Column(
                elem_classes="card"
            ):

                gr.Markdown(
                    "### Enhanced Image"
                )

                enhanced_result = gr.Image(
                    type="pil",
                    label="Enhanced",
                    interactive=False
                )

        another_btn = gr.Button(
            "ENHANCE ANOTHER IMAGE",
            variant="primary"
        )


    # =====================================================
    # NAVIGATION FUNCTIONS
    # =====================================================

    def open_upload():

        return (
            gr.update(visible=False),
            gr.update(visible=True)
        )


    def back_to_welcome():

        return (
            gr.update(visible=True),
            gr.update(visible=False)
        )


    def open_instruction(image):

        return (
            gr.update(visible=False),
            gr.update(visible=True),
            image,
            image
        )


    def back_to_upload():

        return (
            gr.update(visible=True),
            gr.update(visible=False)
        )


    def open_processing():

        return (
            gr.update(visible=False),
            gr.update(visible=True)
        )


    def open_result(original, enhanced):

        return (
            gr.update(visible=False),
            gr.update(visible=True),
            original,
            enhanced
        )


    def restart():

        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            None,
            None,
            "",
            None
        )


    def cancel_application():

        return """
        <div style="text-align:center; padding:20px;">
            <h3>Thank you for using Smart Image Enhancement.</h3>
            <p>You can now close this browser tab.</p>
        </div>
        """


    # =====================================================
    # START BUTTON
    # =====================================================

    start_btn.click(
        fn=open_upload,
        outputs=[
            welcome_page,
            upload_page
        ]
    )


    # =====================================================
    # CANCEL BUTTON
    # =====================================================

    cancel_btn.click(
        fn=cancel_application,
        outputs=exit_message
    )


    # =====================================================
    # UPLOAD → INSTRUCTION
    # =====================================================

    upload_next_btn.click(
        fn=open_instruction,
        inputs=image_input,
        outputs=[
            upload_page,
            instruction_page,
            image_preview,
            object_editor
        ]
    )


    # =====================================================
    # UPLOAD BACK BUTTON
    # =====================================================

    upload_back_btn.click(
        fn=back_to_welcome,
        outputs=[
            upload_page,
            welcome_page
        ]
    )


    # =====================================================
    # INSTRUCTION BACK BUTTON
    # =====================================================

    instruction_back_btn.click(
        fn=back_to_upload,
        outputs=[
            upload_page,
            instruction_page
        ]
    )


    # =====================================================
    # PROMPT SUGGESTIONS
    # =====================================================

    noise_btn.click(
        lambda: "Remove the noise from the image.",
        outputs=prompt_input
    )

    blur_btn.click(
        lambda: "Remove the blur from the image.",
        outputs=prompt_input
    )

    rain_btn.click(
        lambda: "Remove the raindrops from the image.",
        outputs=prompt_input
    )

    haze_btn.click(
        lambda: "Remove the haze from the image.",
        outputs=prompt_input
    )

    light_btn.click(
        lambda: "Improve the lighting of the image.",
        outputs=prompt_input
    )


    # =====================================================
    # NORMAL ENHANCEMENT
    # =====================================================

    enhance_btn.click(
        fn=open_processing,
        outputs=[
            instruction_page,
            processing_page
        ]
    ).then(
        fn=process_img,
        inputs=[
            image_input,
            prompt_input
        ],
        outputs=[
            enhanced_result
        ]
    ).then(
        fn=open_result,
        inputs=[
            image_input,
            enhanced_result
        ],
        outputs=[
            processing_page,
            result_page,
            original_result,
            enhanced_result
        ]
    )


    # =====================================================
    # OBJECT REMOVAL
    # =====================================================

    remove_object_btn.click(
        fn=remove_selected_object,
        inputs=[
            object_editor
        ],
        outputs=[
            enhanced_result
        ]
    ).then(
        fn=open_result,
        inputs=[
            image_input,
            enhanced_result
        ],
        outputs=[
            processing_page,
            result_page,
            original_result,
            enhanced_result
        ]
    )


    # =====================================================
    # ENHANCE ANOTHER IMAGE
    # =====================================================

    another_btn.click(
        fn=restart,
        outputs=[
            welcome_page,
            upload_page,
            result_page,
            image_input,
            image_preview,
            prompt_input,
            object_editor
        ]
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    demo.launch(
        inbrowser=True
    )
import http.server
import socketserver
from urllib.parse import urlparse
from urllib.parse import parse_qs
import tempfile
from pathlib import Path

from backend.lcm_text_to_image import LCMTextToImage
from backend.models.lcmdiffusion_setting import LCMLora, LCMDiffusionSetting
from constants import DEVICE
from time import perf_counter
import numpy as np
from cv2 import imencode

from PIL import Image
import onnxruntime
from huggingface_hub import hf_hub_download

import io

lcm_text_to_image = LCMTextToImage()
lcm_lora = LCMLora(
    base_model_id="Lykon/dreamshaper-8",
    lcm_lora_id="latent-consistency/lcm-lora-sdv1-5",
)

lastimagefile = Path(tempfile.gettempdir()) / 'web-slideshow-last-image.png'
print('Last image file name: ' + str(lastimagefile))

class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urlparse(self.path)
        if parsed_url.path == '/':
            response = '''<!DOCTYPE html>
<html lang="en">
    <head>
        <title>SD Slideshow</title>
        <script type="text/javascript">
            function showLastImage() {
                fetch("last.png").then(async (response) => {
                    if (response.ok) {
                        const blobData = await response.blob();
                        document.getElementById("mainimage").src = URL.createObjectURL(blobData);
                    }
                });
            }
            function showNewImage() {
                const prompts = [
                    "Verdant valley with wildflowers, winding path, oil painting, impasto",
                    "Cyberpunk cityscape, neon lights, skyscrapers, 4k",
                    "Southwest scene with beautiful towering mesas, 4k",
                ];
                const selectedPrompt = prompts[Math.floor(Math.random() * prompts.length)];
                fetch(`img.png?prompt=${encodeURIComponent(selectedPrompt)}`).then(async (response) => {
                    if (response.ok) {
                        const blobData = await response.blob();
                        document.getElementById("mainimage").src = URL.createObjectURL(blobData);
                    }
                });
            }
            showLastImage();
            showNewImage();
            setInterval(showNewImage, 600000);
        </script>
    </head>
    <body style="background-color: black; display: flex; align-items: center; justify-content: center; width: 100vw; height: 100vh; margin: 0; overflow: hidden; cursor: none;">
        <img id="mainimage" style="height: 100%;" />
    </body>
</html>
'''
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(bytes(response, "utf8"))
            return
        if parsed_url.path == '/last.png' and lastimagefile.is_file():
            with lastimagefile.open('rb') as f:
                byte_data = f.read()
                self.send_response(200)
                self.send_header("Content-type", "image/png")
                self.end_headers()
                self.wfile.write(byte_data)
                return
        if parsed_url.path != '/img.png':
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-type", "image/png")
        self.end_headers()

        prompt = "Verdant valley with wildflowers, oil painting, impasto"
        steps = 4
        query_components = parse_qs(parsed_url.query)
        if "prompt" in query_components:
            prompt = query_components["prompt"][0]
        if "steps" in query_components:
            steps = int(query_components["steps"][0])

        lcm_diffusion_setting = LCMDiffusionSetting()
        lcm_diffusion_setting.use_offline_model = True
        lcm_diffusion_setting.use_safety_checker = True
        lcm_diffusion_setting.use_tiny_auto_encoder = False
        lcm_diffusion_setting.openvino_lcm_model_id = "rupeshs/LCM-dreamshaper-v7-openvino"
        lcm_diffusion_setting.prompt = prompt
        lcm_diffusion_setting.negative_prompt = "ugly, deformed, duplicate, frame"
        lcm_diffusion_setting.guidance_scale = 1.5
        lcm_diffusion_setting.inference_steps = steps
        lcm_diffusion_setting.seed = 0
        lcm_diffusion_setting.use_seed = False
        lcm_diffusion_setting.image_width = 512
        lcm_diffusion_setting.image_height = 256
        lcm_diffusion_setting.use_openvino = True
        lcm_text_to_image.init(
            DEVICE,
            lcm_diffusion_setting,
        )
        start = perf_counter()

        images = lcm_text_to_image.generate(lcm_diffusion_setting)
        latency = perf_counter() - start
        print(f"Latency: {latency:.2f} seconds")
        image_arr = np.asarray(images[0])[:, :, ::-1]
        _, byte_data = imencode(".png", image_arr)

        input_image = Image.open(io.BytesIO(byte_data)).convert("RGB")
        input_image = np.array(input_image).astype("float32")
        input_image = np.transpose(input_image, (2, 0, 1))
        img_arr = np.expand_dims(input_image, axis=0)

        if np.max(img_arr) > 256:  # 16-bit image
            max_range = 65535
        else:
            max_range = 255.0
            img = img_arr / max_range

        model_path = hf_hub_download(
            repo_id="rupeshs/edsr-onnx",
            filename="edsr_onnxsim_2x.onnx",
        )
        sess = onnxruntime.InferenceSession(model_path)

        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        output = sess.run(
            [output_name],
            {input_name: img},
        )[0]

        result = output.squeeze()
        result = result.clip(0, 1)
        image_array = np.transpose(result, (1, 2, 0))
        image_array = np.uint8(image_array * 255)
        upscaled_image = Image.fromarray(image_array)

        image_arr = np.asarray(upscaled_image)[:, :, ::-1]
        _, byte_data = imencode(".png", image_arr)

        with lastimagefile.open('wb') as f:
            f.write(byte_data)

        self.wfile.write(byte_data)

        return

handler_object = MyHttpRequestHandler

my_server = socketserver.TCPServer(("", 8000), handler_object)
my_server.serve_forever()

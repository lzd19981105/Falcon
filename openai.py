import argparse
import re
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, List, Literal, Optional, Union,Any
from pydantic import BaseModel, Field
import time
import cv2
import base64

@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    # content: Optional[Union[str, list]]
    content: Optional[Union[str,List[Any]]]
    function_call: Optional[Dict] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    functions: Optional[List[Dict]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[
        Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
    ]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))

class CoordinatesQuantizer:
    def __init__(self, mode, bins):
        self.mode = mode
        self.bins = bins

    def quantize(self, coordinates: torch.Tensor, size):
        bins_w, bins_h = self.bins
        size_w, size_h = size
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        assert coordinates.shape[-1] == 2, "coordinates should be shape (N, 2)"
        x, y = coordinates.split(1, dim=-1)
        if self.mode == "floor":
            quantized_x = (x / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_y = (y / size_per_bin_h).floor().clamp(0, bins_h - 1)
        else:
            raise ValueError("Incorrect quantization type.")
        return torch.cat((quantized_x.int(), quantized_y.int()), dim=-1)

    def dequantize(self, coordinates: torch.Tensor, size):
        bins_w, bins_h = self.bins
        size_w, size_h = size
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        x, y = coordinates.split(1, dim=-1)
        if self.mode == "floor":
            dequantized_x = (x + 0.5) * size_per_bin_w
            dequantized_y = (y + 0.5) * size_per_bin_h
        return torch.cat((dequantized_x, dequantized_y), dim=-1)

class BoxQuantizer(object):
    def __init__(self, mode, bins):
        self.mode = mode
        self.bins = bins

    def quantize(self, boxes: torch.Tensor, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size       # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        xmin, ymin, xmax, ymax = boxes.split(1, dim=-1)  # Shape: 4 * [N, 1].

        if self.mode == 'floor':
            quantized_xmin = (
                xmin / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_ymin = (
                ymin / size_per_bin_h).floor().clamp(0, bins_h - 1)
            quantized_xmax = (
                xmax / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_ymax = (
                ymax / size_per_bin_h).floor().clamp(0, bins_h - 1)

        elif self.mode == 'round':
            raise NotImplementedError()

        else:
            raise ValueError('Incorrect quantization type.')

        quantized_boxes = torch.cat(
            (quantized_xmin, quantized_ymin, quantized_xmax, quantized_ymax), dim=-1
        ).int()

        return quantized_boxes

    def dequantize(self, boxes: torch.Tensor, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size       # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        xmin, ymin, xmax, ymax = boxes.split(1, dim=-1)  # Shape: 4 * [N, 1].

        if self.mode == 'floor':
            # Add 0.5 to use the center position of the bin as the coordinate.
            dequantized_xmin = (xmin + 0.5) * size_per_bin_w
            dequantized_ymin = (ymin + 0.5) * size_per_bin_h
            dequantized_xmax = (xmax + 0.5) * size_per_bin_w
            dequantized_ymax = (ymax + 0.5) * size_per_bin_h

        elif self.mode == 'round':
            raise NotImplementedError()

        else:
            raise ValueError('Incorrect quantization type.')

        dequantized_boxes = torch.cat(
            (dequantized_xmin, dequantized_ymin,
             dequantized_xmax, dequantized_ymax), dim=-1
        )

        return dequantized_boxes

_TEXT_COMPLETION_CMD = object()

def convert_base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str.split(';base64,')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def extract_polygons(generated_text, image_size, coordinates_quantizer):
    pattern = r"<poly>(.*?</poly>)"
    matches = re.findall(pattern, generated_text, re.DOTALL)
    polygons = []
    for polygon_str in matches:
        coords = re.findall(r"<(\d+)>", polygon_str)
        coords = list(map(int, coords))
        if len(coords) % 2 != 0:
            coords = coords[:-1]
        coords_tensor = torch.tensor(coords).view(-1, 2)
        dequantized = coordinates_quantizer.dequantize(coords_tensor, image_size).view(-1).tolist()
        polygons.append(dequantized)
    return polygons

def extract_roi(input_string, pattern=r"<(\d+)><(\d+)><(\d+)><(\d+)>"):
    matches = re.findall(pattern, input_string)
    return [list(map(int, match)) for match in matches]

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="Falcon_Large")
    return ModelList(data=[model_card])

def extract_prompt(input_string,split_token=":"):
    try:
        post_process_type,prompt = input_string.split(split_token)
        
    except:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: Expecting user input.",
        )
    return post_process_type,prompt

def post_process_prompt(post_process_type,prompt,image):
    bboxes = None
    if post_process_type in ["REG_CLS_HBB","REG_CAP"]:
        bboxes = re.findall(r"\d+", prompt)
        bboxes = [int(num) for num in bboxes]
        assert len(bboxes) % 4 == 0, "Invalid bbox format"
        x1,y1,x2,y2 = bbox_quantizer.quantize(torch.tensor(np.array(bboxes).reshape(1, 4)),(image.width, image.height)).view(-1).tolist()
        prompt = re.sub(r"\[.+\]", f"<box><{x1}><{y1}><{x2}><{y2}></box>", prompt)
    elif post_process_type in ["REG_CLS_OBB"]:
        bboxes = re.findall(r"[-+]?\d*\.?\d+", prompt)
        bboxes = [int(float(num)) for num in bboxes]
        assert len(bboxes) % 8 == 0, "Invalid bbox format"
        x1,y1,x2,y2,x3,y3,x4,y4 = coordinates_quantizer.quantize(torch.tensor(np.array(bboxes).reshape(4, 2)),(image.width, image.height)).view(-1).tolist()
        prompt = re.sub(r"\[.+\]", f"<quad><{x1}><{y1}><{x2}><{y2}><{x3}><{y3}><{x4}><{y4}></quad>", prompt)
    return prompt,bboxes

def parse_messages(messages):
    if messages[-1].role != "user":
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: Expecting at least one user message.",
        )
    image1,image2,prompt,post_process_type = None, None, None, None
    prompt = _TEXT_COMPLETION_CMD
    if messages[-1].role == "user":
        if isinstance(messages[-1].content,list):
            post_process_type,prompt = extract_prompt(messages[-1].content[0]["text"])
            image1 = convert_base64_to_image(messages[-1].content[1]["image_url"]["url"])
            if len(messages[-1].content) > 2:
                image2 = convert_base64_to_image(messages[-1].content[2]["image_url"]["url"])
        else:
            post_process_type,prompt = extract_prompt(messages[-1].content)
    # 得是奇数
    if len(messages) % 2 == 0:
        raise HTTPException(status_code=400, detail="Invalid request")
    
    messages = messages[:-1]  # 去掉最后一个用户消息
    # 在中最好只保存最近的一个图片，以防止进行多轮问答
    if image1 is None:
        for i in range(len(messages)-1, -1, -2):
            if messages[i-1].role == "user" and messages[i].role == "assistant":
                if isinstance(messages[i-1].content,list):
                    image1 = convert_base64_to_image(messages[i-1].content[1]["image_url"]["url"])
                    if len(messages[i-1].content) > 2:
                        image2 = convert_base64_to_image(messages[i-1].content[2]["image_url"]["url"])
                    break
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid request: Expecting exactly one user (or function) role before every assistant role.",
                )    
    return image1,image2,prompt.strip(),post_process_type.strip()

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, processor,coordinates_quantizer
    image1,image2,prompt,post_process_type = parse_messages(request.messages)
    image = Image.fromarray(np.uint8(image1))
    if image2 is not None:
        img = np.zeros(image1.shape)
        half1 = np.concatenate((image1, img), axis=0)
        half2 = np.concatenate((img, image2), axis=0)
        image = np.concatenate((half1, half2), axis=1)
        image = Image.fromarray(np.uint8(image))
    prompt,bboxes = post_process_prompt(post_process_type,prompt,image)
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"].to(model.device),
        pixel_values=inputs["pixel_values"].to(model.device),
        max_new_tokens=8192,
        num_beams=3,
        do_sample=False,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    if post_process_type in [
        "REG_VG",
        "REG_DET_HBB",
        "REG_DET_OBB",
        "PIX_SEG",
        'PIX_CHG'
    ]:
        if post_process_type == "REG_DET_OBB":     # obb detection
            pred_bboxes = extract_roi(
                generated_text,
                pattern=r"<(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)>",
            )
            answer = []
            for bbox in pred_bboxes:
                quad_box = coordinates_quantizer.dequantize(
                            torch.tensor(np.array(bbox).reshape(4, 2)),
                            size=(image.width, image.height)
                        ).view(-1).tolist()
                answer.append(list(quad_box))
        elif post_process_type in ["REG_VG", "REG_DET_HBB"]:  # hbb detection
            pred_bboxes = extract_roi(
                generated_text, pattern=r"<(\d+)><(\d+)><(\d+)><(\d+)>"
            )
            answer = []
            for bbox in pred_bboxes:
                bbox = bbox_quantizer.dequantize(
                            torch.tensor(np.array(bbox).reshape(1, 4)),
                            size=(image.width, image.height)
                        ).view(-1).tolist()
                answer.append(list(bbox))
        elif post_process_type == "PIX_SEG" or post_process_type == 'PIX_CHG':  # segmentation and change detection
            pred_polygons = extract_polygons(
                generated_text, (image.width, image.height), coordinates_quantizer
            )
            answer = pred_polygons
        else:
            print("Unknown task ", post_process_type)
        answer = "```"+ str(answer) +"```"
    elif post_process_type in ["REG_CLS_HBB", "REG_CLS_OBB", "REG_CAP"]:
        answer = generated_text.replace("</s>", "").replace("<s>", "").replace('<pad>','').replace('</pad>','')
        answer = answer + "```[" + str(bboxes) + "]```"
    else:
        answer = generated_text.replace("</s>", "").replace("<s>", "").replace('<pad>','').replace('</pad>','')
    

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=answer),
        finish_reason="stop",
    )
    return ChatCompletionResponse(
        model=request.model, choices=[choice_data], object="chat.completion"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="checkpoint path")
    args = parser.parse_args()  # 需要提前设置检查点路径
    
    coordinates_quantizer = CoordinatesQuantizer(mode="floor", bins=(1000, 1000))
    bbox_quantizer = BoxQuantizer(mode="floor", bins=(1000, 1000))
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
    )
    uvicorn.run(app, host="0.0.0.0", port=10086,reload=True)
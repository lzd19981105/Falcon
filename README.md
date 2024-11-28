<div align="center">

# <img width="60" alt="image" src="assets/falcon.png"> Falcon: A Remote Sensing Vision-Language Foundation Model

<div align="center">
  <img width="500" alt="image" src="assets/tianhui.png">
  <br>
</div>

[\[🚀 Quick Start\]](#quick-start-with-Falcon)



<img height="55" alt="image" src="https://github.com/user-attachments/assets/bd62ab46-f0ea-40c6-ab10-7fde671716cc">

![opencompass](assets/radar_graph.png)

</div>

## News 🚀🚀🚀

- `2024/11/27`: Falcon has been released. The model checkpoints is now available on HuggingFace, and both training / evaluation data and scripts are open-sourced.


## Model Zoo

<table>
  <tr>
    <th>Model Name</th>
    <th>HF&nbsp;Link</th>
    <th>MS&nbsp;Link</th>
  </tr>
  <tr>
    <td>Falcon-Single-Instruction-0.7B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2-1B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2-1B">🤖 link</a></td>
  </tr>
  <tr>
    <td>Falcon-Multi-Instruction-0.7B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2-1B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2-1B">🤖 link</a></td>
  </tr>
  <tr>
    <td>Falcon-Single-Instruction-0.3B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2-1B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2-1B">🤖 link</a></td>
  </tr>
</table>

## What can Falcon do?
![opencompass](assets/task_example.png)

## Quick Start with Falcon

<details>
  <summary>Installation (click to expand)</summary>

```bash
conda create -n falon python=3.10
conda activate falcon
pip install -r requirements.txt
```

</details>

<details>
  <summary>Datasets preperation (click to expand)</summary>

Download FCD-78M dataset which can be found in [here](https://modelscope.cn/models/OpenGVLab/InternVL2-1B). Then, unzip and place/link the dataset at the root path of this repo. The directory structure should be as follows:
```bash
|-Datasets
|----XXX_task
|    |---xxxxx.jpg
|    ...
|----XXX_task
|    |---xxxxx.jpg
|    ...
|----train_label
|    |---single_instruction_conversation_train.json
|    |---multi_instruction_conversation_train.json
|    ...
|----test_label
|    |---XXX_test.json
|    ...
```

</details>

<details>
  <summary>Training Falcon with FCD-78M (click to expand)</summary>

1. Download the checkpoints you want and place them at the root path of this repo. The directory structure should be as follows:
```bash
|-model_checkpoints
|----Falcon-Single-Instruction-0.7B
|    |---pytorch_model.bin
|    ...
|----Falcon-Multi-Instruction-0.7B
|    |---pytorch_model.bin
|    ...
|...
```

2. Here we give an example of a training script used for single instruction training. You may runing this script on master machine node and every slave machine node you have. Note that some parameters in this script should be modified according to the machine node on which it is running.

```bash
RANK=0 # The node idx of current machine node
WORLD_SIZE=1 # The total number of machine node
GPU_NUM=8 # The number of gpu in each machine node
MASTER_ADDR=localhost # The IP address of the master machine node
MASTER_PORT=12355 # The port of the master machine node

python multi_node_distributed_train.py \
    --node_rank $RANK \
    --world_size $(($GPU_NUM*$WORLD_SIZE)) \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --checkpoint_path model_checkpoints/<checkpoint_dir_name> \
    --dataset FCD-78M \
    --label_json Datasets/train_label/single_instruction_conversation_train.json \
    --num_workers 2 \
    --batch_size 7 \  # Adjust this value according to your GPU memory
    --epochs 3 \
    --run_name Falcon-Single-Instruction-0.7B_new
```
</details>

<details>
  <summary>Evaluating Falcon with FCD-78M (click to expand)</summary>

1. Here we provide an example of the evaluation program to evaluate Falcon using FCD-78M dataset with the json annotation file.

```bash
GPU=0
CUDA_VISIBLE_DEVICES=$GPU python single_gpu_inference_eval.py \
    --model-path model_checkpoints/<checkpoint_dir_name> \
    --eval-file Datasets/test_label/single_instruction_conversation_test.json \
    --dataset-path FCD-78M \
    --result-path ./ \
    --batch_size 8 \  # Adjust this value according to your GPU memory
    --num_workers 2 \
```

2. To calculate the evaluation metrics, please follow this bash format.

```bash
python evaluation.py
```

</details>

## License

This project is released under the [MIT license](LICENSE). Parts of this project contain code and models from other sources, which are subject to their respective licenses.

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@article{yao2025falcon,
  title={Falcon: A Remote Sensing Vision-Language Foundation Model},
  author={kelu, Yao and Nuo, Xu and Rong, Yang and Yingying, Xu and Titinunt, Kitrungrotsakul and Zhuoyan, Gao and yi, Ren and Jin, Wang and Ning, Wei and Chao, Li},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Acknowledgement

Falcon is built with reference to the code of the following projects: [Florence-2-base-ft](https://huggingface.co/microsoft/Florence-2-base-ft), [Florence-2-large-ft](https://huggingface.co/microsoft/Florence-2-large-ft), [florence2-finetuning](https://github.com/andimarafioti/florence2-finetuning). Thanks for their awesome work!

#!/usr/bin/env python3
import os
import sys
import json
import struct
import numpy as np
import torch
from sentencepiece import SentencePieceProcessor
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """模型配置类"""
    vocab_size: int
    dim: int
    multiple_of: int
    n_heads: int
    n_layers: int
    norm_eps: float = 1e-6  # 添加norm_eps参数
    ftype: int = 1  # 默认使用float16

    @classmethod
    def from_json(cls, json_path: str, tokenizer) -> 'ModelConfig':
        """从JSON文件加载配置"""
        with open(json_path, "r") as f:
            params = json.load(f)
        
        # 确保所有必要的参数都存在
        required_params = {
            "dim": int,
            "multiple_of": int,
            "n_heads": int,
            "n_layers": int,
            "norm_eps": float
        }
        
        # 验证参数
        for param, param_type in required_params.items():
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")
            try:
                params[param] = param_type(params[param])
            except (ValueError, TypeError):
                raise ValueError(f"Invalid type for parameter {param}")
        
        # 添加词汇表大小
        params["vocab_size"] = tokenizer.vocab_size()
        
        return cls(**params)

    def get_n_parts(self) -> int:
        """根据维度确定模型分片数"""
        dim_to_parts = {
            4096: 1,
            5120: 2,
            6656: 4,
            8192: 8
        }
        if self.dim not in dim_to_parts:
            raise ValueError(f"Invalid dimension: {self.dim}")
        return dim_to_parts[self.dim]

class Tokenizer:
    """分词器包装类"""
    def __init__(self, model_path: str):
        self.sp = SentencePieceProcessor(model_path)

    def vocab_size(self) -> int:
        return self.sp.vocab_size()

    def is_unknown(self, token_id: int) -> bool:
        return self.sp.is_unknown(token_id)

    def is_control(self, token_id: int) -> bool:
        return self.sp.is_control(token_id)

    def is_byte(self, token_id: int) -> bool:
        return self.sp.is_byte(token_id)

    def id_to_piece(self, token_id: int) -> str:
        return self.sp.id_to_piece(token_id)

class ModelConverter:
    """模型转换器类"""
    FTYPE_STR = ["f32", "f16"]
    MAGIC = 0x67677d6c

    def __init__(self, model_dir: str, ftype: int = 1):
        self.model_dir = model_dir
        self.ftype = ftype
        self.tokenizer = Tokenizer(os.path.join(model_dir, "tokenizer.model"))
        self.config = ModelConfig.from_json(
            os.path.join(model_dir, "params.json"),
            self.tokenizer
        )
        self.config.ftype = ftype

    def _write_header(self, fout) -> None:
        """写入模型文件头"""
        fout.write(struct.pack("i", self.MAGIC))
        fout.write(struct.pack("i", self.config.vocab_size))
        fout.write(struct.pack("i", self.config.dim))
        fout.write(struct.pack("i", self.config.multiple_of))
        fout.write(struct.pack("i", self.config.n_heads))
        fout.write(struct.pack("i", self.config.n_layers))
        fout.write(struct.pack("i", self.config.dim // self.config.n_heads))
        fout.write(struct.pack("i", self.ftype))

    def _write_vocab(self, fout) -> None:
        """写入词汇表"""
        for i in range(self.tokenizer.vocab_size()):
            if self.tokenizer.is_unknown(i):
                text = " \u2047 ".encode("utf-8")
                fout.write(struct.pack("i", len(text)))
                fout.write(text)
            elif self.tokenizer.is_control(i):
                fout.write(struct.pack("i", 0))
            elif self.tokenizer.is_byte(i):
                piece = self.tokenizer.id_to_piece(i)
                if len(piece) != 6:
                    raise ValueError(f"Invalid token: {piece}")
                byte_value = int(piece[3:-1], 16)
                fout.write(struct.pack("i", 1))
                fout.write(struct.pack("B", byte_value))
            else:
                text = self.tokenizer.id_to_piece(i).replace("\u2581", " ").encode("utf-8")
                fout.write(struct.pack("i", len(text)))
                fout.write(text)

    def _write_tensor(self, fout, name: str, data: np.ndarray) -> None:
        """写入张量数据"""
        if name.endswith("freqs"):
            return

        print(f"Processing variable: {name} with shape: {data.shape} and type: {data.dtype}")
        
        n_dims = len(data.shape)
        dshape = data.shape

        # 确定数据类型
        ftype_cur = 1
        if self.ftype == 0 or n_dims == 1:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0

        # 写入头部信息
        sname = name.encode('utf-8')
        fout.write(struct.pack("iii", n_dims, len(sname), ftype_cur))
        for i in range(n_dims):
            fout.write(struct.pack("i", dshape[n_dims - 1 - i]))
        fout.write(sname)

        # 写入数据
        data.tofile(fout)

    def convert_part(self, part: int) -> str:
        """转换单个模型分片"""
        model_path = os.path.join(self.model_dir, f"consolidated.0{part}.pth")
        output_path = os.path.join(self.model_dir, f"dragon-model-{self.FTYPE_STR[self.ftype]}.bin")
        if part > 0:
            output_path += f".{part}"

        if os.path.exists(output_path):
            print(f"Skip conversion, it already exists: {output_path}")
            return output_path

        print(f'Processing part {part}')
        model = torch.load(model_path, map_location="cpu")

        with open(output_path, "wb") as fout:
            self._write_header(fout)
            self._write_vocab(fout)

            for name, tensor in model.items():
                self._write_tensor(fout, name, tensor.numpy().squeeze())

        print(f"Done. Output file: {output_path}, (part {part})")
        return output_path

    def convert(self) -> None:
        """转换整个模型"""
        n_parts = self.config.get_n_parts()
        for part in range(n_parts):
            self.convert_part(part)

def main():
    if len(sys.argv) < 3:
        print("Usage: convert-ckpt-to-dragon.py dir-model ftype\n")
        print("  ftype == 0 -> float32")
        print("  ftype == 1 -> float16")
        sys.exit(1)

    model_dir = sys.argv[1]
    ftype = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    if ftype < 0 or ftype > 1:
        print(f"Invalid ftype: {ftype}")
        sys.exit(1)

    try:
        converter = ModelConverter(model_dir, ftype)
        converter.convert()
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

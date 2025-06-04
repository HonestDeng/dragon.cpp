#!/usr/bin/env python3
import os
from sentencepiece import SentencePieceProcessor

def read_control_tokens(model_path: str):
    """读取tokenizer.model中的所有控制token"""
    sp = SentencePieceProcessor(model_path)
    print(sp.vocab_size())
    
    print("控制token列表：")
    print("-" * 50)
    print(f"{'Token ID':<10} {'Token':<20} {'Piece':<30}")
    print("-" * 50)
    
    for i in range(sp.vocab_size()):
        if sp.is_control(i):
            piece = sp.id_to_piece(i)
            print(f"{i:<10} {piece:<20} {sp.decode([i]):<30}")

def main():
    if len(sys.argv) != 2:
        print("使用方法: python read_control_tokens.py <tokenizer.model路径>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"错误：找不到文件 {model_path}")
        sys.exit(1)
        
    try:
        read_control_tokens(model_path)
    except Exception as e:
        print(f"处理过程中出错：{e}")
        sys.exit(1)

if __name__ == "__main__":
    import sys
    main() 
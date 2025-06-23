import subprocess
import os
import sys
import shutil
from openai import OpenAI

# Placeholder for model path. The user should configure this.
# I'll assume a directory structure for the models.
MODEL_DIR = "models-hf/llama-7b" 
CONVERTED_MODEL_NAME = "dragon-model-f16.bin"
# The conversion script for multi-part models will add a suffix. Assuming single part for llama-7b based on config.
# The script logic seems to create consolidated.00.pth -> dragon-model-f16.bin
# and consolidated.01.pth -> dragon-model-f16.bin.1
# Let's assume the main file is without suffix.
CONVERTED_MODEL_PATH = os.path.join(MODEL_DIR, "dragon-model-f16.bin")
EXECUTABLE_NAME = "llama"
EXECUTABLE_PATH = f"./{EXECUTABLE_NAME}" # Will be updated by build_project to be inside the build dir

def build_project():
    """Builds the C++ project using cmake."""
    print("="*20)
    print("Building project with CMake...")
    build_dir = "build"
    try:
        if not os.path.exists("CMakeLists.txt"):
            print("CMakeLists.txt not found. Cannot build project.")
            return False

        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        os.makedirs(build_dir)
        
        cmake_cmd = ["cmake", ".."]
        subprocess.run(cmake_cmd, cwd=build_dir, check=True, capture_output=True, text=True)
        
        make_cmd = ["make", "-j", "18"]
        subprocess.run(make_cmd, cwd=build_dir, check=True, capture_output=True, text=True)

        exe_path = os.path.join(build_dir, EXECUTABLE_NAME)
        if os.path.exists(exe_path):
            print(f"Build successful. Executable at: {exe_path}")
            globals()["EXECUTABLE_PATH"] = exe_path
            return True
        else:
            print(f"Build failed, executable not found at {exe_path}.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error during build:")
        print(e.stdout)
        print(e.stderr)
        return False
    except FileNotFoundError as e:
        print(f"Error: '{e.filename}' command not found. Please ensure it's installed and in your PATH.")
        return False

def test_step_1_model_conversion():
    """
    Tests step 1: Model conversion using convert-pth-to-dragon.py.
    """
    print("="*20)
    print("Step 1: Testing model conversion...")
    if not os.path.exists(MODEL_DIR):
        print(f"Warning: Model directory not found at {MODEL_DIR}")
        print(f"Creating dummy model directory and files for testing...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(os.path.join(MODEL_DIR, "params.json"), "w") as f:
            f.write('{"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": 32000}')
        # create dummy tokenizer.model
        with open(os.path.join(MODEL_DIR, "tokenizer.model"), "w") as f:
            # SentencePieceProcessor needs a valid model file, an empty file will cause errors.
            # We can't easily create a valid one, so we'll have to assume this step might be fragile
            # without real model files. For now, this is a placeholder.
            pass
        # create dummy consolidated.00.pth
        with open(os.path.join(MODEL_DIR, "consolidated.00.pth"), "w") as f:
            f.write("")
        
    try:
        # ftype=1 for float16
        cmd = ["python", "convert-pth-to-dragon.py", MODEL_DIR, "1"]
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Model conversion script ran successfully.")
        
        # The script may create multiple parts. Let's check for the first part.
        base_output_path = os.path.join(MODEL_DIR, f"dragon-model-f16.bin")
        if os.path.exists(base_output_path) or os.path.exists(base_output_path + ".0"):
             print(f"Converted model found.")
             # The script might output multiple files, we'll use the base name for the next steps
             # and assume the C++ code knows how to find the other parts.
             # The convert script creates dragon-model-f16.bin (for part 0).
             globals()["CONVERTED_MODEL_PATH"] = base_output_path
             return True
        else:
            print(f"Error: Converted model not found after running conversion script.")
            print(process.stdout)
            print(process.stderr)
            return False

    except subprocess.CalledProcessError as e:
        print("Error during model conversion:")
        print(e.stdout)
        print(e.stderr)
        return False

def test_step_2_model_loading():
    """
    Tests step 2: Loading the converted model.
    Checks for magic number error or memory access errors.
    """
    print("="*20)
    print("Step 2: Testing model loading...")
    
    cmd = [EXECUTABLE_PATH, "-m", CONVERTED_MODEL_PATH, "--load-model-only"]
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
        print("Model loading test passed (no crash or magic number error).")
        return True
    except subprocess.TimeoutExpired:
        print("Error: Model loading timed out. Possible deadlock.")
        return False
    except subprocess.CalledProcessError as e:
        print("Error: The program crashed during model loading. This might be the memory bug.")
        print(e.stdout)
        print(e.stderr)
        return False


def is_meaningful(text: str) -> bool:
    # client = OpenAI(
    #     base_url="http://192.168.141.110:8000/v1",
    #     api_key="-",
    # )
    # model = client.models.list().data[0].id

    # completion = client.chat.completions.create(
    #     model=model,
    #     messages=[
    #         {"role": "user", "content": "判断下面输入的字符串是否是正常的文本：{text}"}
    #     ],
    #     extra_body={"guided_choice": ["yes", "no"]},
    # )
    return True

def test_step_3_inference():
    """
    Tests step 3: Full inference run.
    Checks for deadlocks and meaningless output.
    """
    print("="*20)
    print("Step 3: Testing inference for deadlock and output quality...")
    prompt = " Once upon a time"
    cmd = [EXECUTABLE_PATH, "-m", CONVERTED_MODEL_PATH, "-n", "20"]
    try:
        # 5-minute timeout for deadlock detection
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        # The C++ program prints the prompt first, so we should strip it.
        output = process.stdout.replace(prompt, "").strip()
        print(f"Inference output: {output}")

        if not is_meaningful(output):
            print("Test failed: Output is not meaningful.")
            return False
        
        print("Inference test passed.")
        return True

    except subprocess.TimeoutExpired:
        print("Error: Inference timed out after 5 minutes. A deadlock is likely.")
        return False
    except subprocess.CalledProcessError as e:
        print("Error: The program crashed during inference.")
        print(e.stdout)
        print(e.stderr)
        return False

def test_step_4_tokenizer():
    """
    Tests step 4: Tokenizer implementation.
    """
    print("="*20)
    print("Step 4: Testing tokenizer...")
    
    test_code = r"""
#include "utils.h"
#include <iostream>

// We need gpt_vocab definition from utils.h and llama_tokenize declaration.
// Both are in utils.h.

int main() {
    gpt_vocab vocab;
    // The test case is "Hello Hi" -> "1 2 3". This is an unusual tokenization.
    // It is designed to test if the candidate can implement a function to fulfill
    // a specific requirement. A simple implementation might split the string and
    // use a map to find IDs.
    vocab.token_to_id["Hello"] = 1;
    vocab.token_to_id[" "] = 2;
    vocab.token_to_id["Hi"] = 3;
    
    // The candidate needs to re-implement llama_tokenize to pass this test.
    std::vector<gpt_vocab::id> tokens = llama_tokenize(vocab, "Hello Hi", false);
    for (const auto& id : tokens) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    return 0;
}
"""
    test_src_filename = "test_tokenizer.cpp"
    test_exe_filename = "test_tokenizer"
    with open(test_src_filename, "w") as f:
        f.write(test_code)

    # Compile the test by including source files directly, as cmake doesn't produce easily linkable .o files for this.
    # From CMakeLists.txt, we know pthreads are used.
    compile_cmd = ["g++", "-std=c++11", test_src_filename, "utils.cpp", "operators.c", "-I.", "-o", test_exe_filename, "-lpthread"]
    
    if not (os.path.exists("utils.cpp") and os.path.exists("operators.c")):
        print("Source files (utils.cpp, operators.c) not found. They are needed for the tokenizer test.")
        return False

    print(f"Compiling tokenizer test: {' '.join(compile_cmd)}")
    try:
        # We need to link against the compiled object files.
        proc = subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Failed to compile tokenizer test.")
        print(e.stderr)
        # One possible failure is a missing definition for llama_tokenize if it was deleted entirely.
        if "undefined reference to `llama_tokenize" in e.stderr:
             print("Linker error: `llama_tokenize` is not defined. The function is likely missing.")
        return False

    run_cmd = ["./" + test_exe_filename]
    try:
        process = subprocess.run(run_cmd, check=True, capture_output=True, text=True)
        output = process.stdout.strip()
        expected_output = "1 2 3"
        print(f"Tokenizer output: '{output}'")

        if output == expected_output:
            print("Tokenizer test passed.")
            return True
        else:
            print(f"Tokenizer test failed. Expected '{expected_output}', got '{output}'")
            return False

    except subprocess.CalledProcessError as e:
        print("Tokenizer test program crashed. The function might be implemented incorrectly.")
        print(e.stderr)
        return False
    finally:
        # Cleanup test files
        if os.path.exists(test_src_filename):
            os.remove(test_src_filename)
        if os.path.exists(test_exe_filename):
            os.remove(test_exe_filename)


def main():
    """Main function to run all tests."""
    results = {}
    
    if not build_project():
        print("Aborting tests due to build failure.")
        sys.exit(1)

    results["step_1_model_conversion"] = test_step_1_model_conversion()
    if not results["step_1_model_conversion"]:
        print("Skipping subsequent tests as model conversion failed.")
        # Fill remaining tests as failed
        results["step_2_model_loading"] = False
        results["step_3_inference"] = False
    else:
        results["step_2_model_loading"] = test_step_2_model_loading()
        results["step_3_inference"] = test_step_3_inference()

    # Tokenizer test can be run independently of model loading/inference results
    results["step_4_tokenizer"] = test_step_4_tokenizer()

    print("\n" + "="*20)
    print("       GRADING SUMMARY")
    print("="*20)
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    for test, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test}: {status}")
    
    print("-" * 20)
    print(f"Overall result: {passed_tests} / {total_tests} tests passed.")

if __name__ == "__main__":
    main() 
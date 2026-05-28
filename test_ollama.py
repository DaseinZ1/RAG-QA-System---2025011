import ollama
import sys

def test_ollama_connection():
    try:
        models = ollama.list()
        if models and len(models.models) > 0:
            print("Ollama连接成功！")
            print(f"已安装的模型:")
            for model in models.models:
                name = getattr(model, 'name', 'unknown')
                size = getattr(model, 'size', 0)
                print(f"  - {name} ({size / (1024**3):.2f} GB)")
            return True
        else:
            print("Ollama已连接，但没有任何模型")
            return False
    except Exception as e:
        print(f"Ollama连接失败: {e}")
        return False

def test_model():
    try:
        print("\n正在测试 deepseek-r1:7b 模型...")
        response = ollama.chat(
            model='deepseek-r1:7b',
            messages=[{'role': 'user', 'content': '请说"测试成功"，只需说这两个字。'}]
        )
        answer = response['message']['content']
        print(f"模型响应: {answer}")
        return True
    except Exception as e:
        print(f"模型测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=== Ollama连接测试 ===\n")
    if test_ollama_connection():
        test_model()
    else:
        print("\n请确保Ollama服务正在运行，并且已下载模型")
        sys.exit(1)

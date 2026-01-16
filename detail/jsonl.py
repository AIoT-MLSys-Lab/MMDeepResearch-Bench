import json
import os

input_file = 'quiz_vef.json'
output_file = 'quiz_vef_clean.jsonl'

# 硬盘上确认是大写 JPG 的文件列表（基于你之前的 dir 输出）
# 只有在 JSON 里是小写但硬盘是的大写时，这个表才起作用，防止 FileNotFoud
KNOWN_UPPERCASE_JPG = {
    "Q12I0", "Q12I1",
    "Q13I2", "Q13I3",
    "Q14I0", "Q14I1", "Q14I2", "Q14I3",
    "Q16I1", "Q16I2",
    "Q18I2", "Q18I3",
    "Q19I2", "Q19I3"
    # Q17I2 在你的列表中是小写 jpg，所以这里不列入
}

def clean_vef_to_jsonl():
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i, item in enumerate(data):
            
            # 1. 修复图片路径
            if "image_url" in item:
                new_images = []
                for img_path in item["image_url"]:
                    # 去掉 'image/' 前缀
                    filename = os.path.basename(img_path)
                    
                    # 检查是否需要修正为 .JPG (大写)
                    name_no_ext = os.path.splitext(filename)[0]
                    if name_no_ext in KNOWN_UPPERCASE_JPG:
                        filename = name_no_ext + ".JPG"
                        
                    new_images.append(filename)
                item["image_url"] = new_images

            # 2. 补充 ID
            item["id"] = i

            # 3. 写入 JSONL
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"转换完成：{output_file}")
    print("已自动移除 image/ 前缀并修正了部分已知的大写后缀问题。")

if __name__ == "__main__":
    clean_vef_to_jsonl()
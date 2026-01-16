import json
import os

# 1. 获取 image 目录下所有文件的真实名字（存入集合方便查找）
# 假设脚本在 image 文件夹的上一级目录运行，如果不是，请修改 path
image_dir = 'image' 
real_filenames = set(os.listdir(image_dir))

# 创建一个 "小写文件名 -> 真实文件名" 的映射字典
# 例如: {'q17i2.jpg': 'Q17I2.jpg', 'q12i0.jpg': 'Q12I0.JPG'}
filename_map = {name.lower(): name for name in real_filenames}

input_jsonl = 'quiz.jsonl'
output_jsonl = 'quiz_final_fixed.jsonl'

with open(input_jsonl, 'r', encoding='utf-8') as f_in, \
     open(output_jsonl, 'w', encoding='utf-8') as f_out:
    
    for line in f_in:
        item = json.loads(line)
        
        if "image_url" in item:
            fixed_images = []
            for img_path in item["image_url"]:
                # img_path 可能是 "Q17I2.JPG"
                # 1. 拿到文件名 (虽然已经是文件名了，保险起见)
                basename = os.path.basename(img_path)
                
                # 2. 转小写去 map 里查真实存在的文件名
                lower_name = basename.lower()
                
                if lower_name in filename_map:
                    # 找到了！用硬盘上真实的文件名替换（比如把 JSON 里的 .JPG 换成硬盘上的 .jpg）
                    real_name = filename_map[lower_name]
                    fixed_images.append(real_name)
                else:
                    # 硬盘上压根没这图
                    print(f"⚠️ 警告: 硬盘上找不到图片 {basename} (忽略大小写也找不到)")
                    fixed_images.append(basename) # 保持原样，或者你可以选择扔掉
            
            item["image_url"] = fixed_images
            
        f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

print("修正完成！quiz_final_fixed.jsonl 里的路径现在和硬盘完全一致了。")
import torch
print(torch.cuda.is_available()) # 沐曦在 MACA 环境下这里通常返回 True
print(torch.cuda.get_device_name(0)) # 应该会显示 MetaX C500
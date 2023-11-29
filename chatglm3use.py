
from modelscope import AutoTokenizer, AutoModel
model_dir = "../chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().quantize(4).cuda()
model = model.eval()
response, history = model.chat(tokenizer, '''将以下JSON转换成 objective-c 视图代码{"id":"0-17004457698920","type":"view","layout":{"overflow":"hidden","width":"375px","height":"200px","display":"block","position":"relative"},"style":{"background-color":"#e91e63"},"children":[{"id":"0-17004457776331","type":"view","layout":{"overflow":"hidden","width":"100px","height":"100px","display":"block","position":"absolute","top":"23px","left":"16px"},"style":{"background-color":"#00CED1"}},{"id":"0-17004494146172","type":"swiper","layout":{"overflow":"hidden","width":"100px","height":"100px","display":"block","position":"absolute","top":"22px","left":"135px"},"style":{"background-color":"#673ab7"}}]}''', history=[])
print(response)
# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# print(response)
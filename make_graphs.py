import json
import matplotlib.pyplot as plt

langs = ['eo', 'tpi', 'fr', 'de', 'ja', 'ru', 'ar', 'es', 'zh']

folder = "models_{lang}-en"

langs = ['ru', 'ar', 'es', 'zh']
bar_data = []
for lang in langs:
    with open(folder.format(lang=lang) + "/val_results.json") as f:
        results = json.load(f)
    bar_data.append(results["val_bleu"])

fig, ax = plt.subplots(figsize=(6, 4))

p = ax.bar(langs, bar_data, label=langs)

ax.bar_label(p, label_type='edge')

ax.set_title('BLEU scores of finetuned models')
ax.set_xlabel('Language Codes')
ax.set_ylabel('BLEU Score')
v = 18.55
ax.plot([ax.get_xlim()[0], ax.get_xlim()[-1]], [v, v],
            ls='--', c='k')
ax.text(ax.get_xlim()[-1]-2, v+0.2, f"Base Model = {v}")
plt.savefig("test.png")
plt.show()


fig, ax = plt.subplots(figsize=(6, 4))

for lang in langs:
    with open(folder.format(lang=lang) + "/trainer_state.json") as f:
        results = json.load(f)
    bleus = []
    for log in results["log_history"]:
        if "eval_bleu" in log:
            bleus.append(log["eval_bleu"])
    ax.plot(list(range(len(bleus))), bleus, label = lang) 
ax.legend() 
ax.set_title('Validation BLEU score during training')
ax.set_xlabel('Epochs')
ax.set_ylabel('BLEU Score')
plt.savefig("test2.png")
plt.show()


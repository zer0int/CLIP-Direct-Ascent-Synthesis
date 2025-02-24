- Technically, a heavily modified fork of:
### [Direct Ascent Synthesis: Hidden Generative Capabilities in Discriminative Models](https://github.com/stanislavfort/Direct_Ascent_Synthesis)
- http://github.com/stanislavfort/Direct_Ascent_Synthesis
- With emphasis on *heavily* modified. Alas, please open an Issue on *me* if you encounter one.
## Like CLIP + VQGAN. Except without a VQGAN.
![banner2](https://github.com/user-attachments/assets/2d64f2fb-51f3-4805-8aae-5bd33d5f755f)
----
----
### ⭐ Update 23-FEB-2025

- Add ability to skip layers in Text & Vision Encoder for generating images
- Counting from the back of the transformer, -1 = last, -2 = penultimate, etc.
- Examples:

Use CLIP-L penultimate (second-to-last) instead of final text encoder layer (like in SDXL!):
```
python clip-generate.py --deterministic --make_anti --manu_vit --manu_txt --model_name "OpenAI-ViT-L/14" --set_vit 1 --set_txt 2
```
- Enable: `--manu_vit` & `--manu_txt` - skip layer (does nothing without Enable): `--set_vit` `--set_txt`
- To also skip final layer normalization before projection: `--skip_ln_vit` and `--skip_ln_txt`
- To reduce batch_size (for VRAM) and augs_cp (quality vs. speed), e.g.: `--batch_size 16` & `--augs_cp 32`
- For all models. Default OpenAI-ViT-B/32: 
- `python clip-generate.py --deterministic --make_anti --manu_vit -set_vit 1 --set_txt 2`

🤖 Also recommended: (layer 20 (of 0-23, vision), layer 11 (of 0-11, text):
```
python clip-generate.py --deterministic --batch_size 16 --augs_cp 32 --make_anti --manu_vit --manu_txt --model_name "OpenAI-ViT-L/14" --set_vit 4 --set_txt 1
```
![cats-compare](https://github.com/user-attachments/assets/cf354db7-6928-4cd6-923c-173d0a683501)
----
### ⭐ First commit 21-FEB-2025

The original author's code offers:

1. Text to image generation
2. "Style" transfer
3. Image reconstruction from its CLIP embedding

This repo adds:

4. Gradient Ascent on the Text Embeddings (use CLIP's own opinion about image as text prompt)
5. Minimize cosine similarity (get an "anti-cat" opinion / antonym for a cat image OR a cat text prompt)
6. Use 4. and 5. to generate images Direct Ascent Synthesis
7. For a given input image, visualize the Neuron (MLP Feature) with the highest activation value.
8. Add one (or all) layer's features / "neurons" to the image stack for processing
9. Can be, in essence, a self-sustained loop of making EVERYTHING out of CLIP. No human input.
10. ...And many more options & features!

![banner1](https://github.com/user-attachments/assets/c3685e3c-af28-4580-889e-c884b20c0966)

Quick start fun, uses human text prompts:

```
python clip-generate.py --use_neuron --make_anti
```

Uses the same default (cat) image as --img0, but gets a CLIP opinion about it (no human text input):
```
python clip-generate.py --use_neuron --make_anti --use_image images/cat.png
```

Adds ALL features ('neurons') as images, changes primary image, changes text prompt 1:
```
python clip-generate.py --all_neurons --img0 images/eatcat.jpg --txt1 "backyardspaghetti lifestyledissertation"
```

Loads a second CLIP model (open_clip), makes plots, CLIP opinion, adds second image --img1:
```
python clip-generate.py --custom_model2 'ViT-B-32' 'laion2b_s34b_b79k' --make_plots --make_lossplots --img1 dogface.png
```

Loads fine-tuned OpenAI/CLIP model as primary, sets deterministic backends. OpenAI models must start with "OpenAI-".
```
python clip-generate.py --model_name "OpenAI-ViT-L/14" "mymodels/finetune.pt" --batch_size 16 --deterministic
```

### Please see code in clip-generate.py for more details.
- There's a lot. But I left you lots of comments, too!
- `python clip-generate.py --help` for a quick review.

![example-of-all](https://github.com/user-attachments/assets/f11ab1e2-898d-4c9b-bc2d-5045aee4a9c1)
-----

## Skip Text Encoder layers until just plugging the first layer into projection - a 1-layer #CLIP text encoder!

- ViT-B/32: *fails*
- ViT-L/14: Relentlessly just makes something else.🦾🤖
- Banana Cat incomprehensible, make: M + 🍟🤡 and 🕑💥🚶🎑🏡⚽️🧦🌟🔢

![relentless-large](https://github.com/user-attachments/assets/f22c079c-2d23-4d1c-a99c-321a9ade00d5)

## A striking difference in complexity for a 12 layer ViT vs. 24 layer ViT:

![comparisons-final](https://github.com/user-attachments/assets/f8f78ca9-2c2c-48c2-9c61-3a3361bf0129)


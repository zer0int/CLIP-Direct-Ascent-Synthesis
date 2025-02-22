- Technically, a heavily modified fork of:
### [Direct Ascent Synthesis: Hidden Generative Capabilities in Discriminative Models](https://github.com/stanislavfort/Direct_Ascent_Synthesis)
- http://github.com/stanislavfort/Direct_Ascent_Synthesis
- With emphasis on *heavily* modified. Alas, please open an Issue on *me* if you encounter one.
## Like CLIP + VQGAN. Except without a VQGAN.
![banner2](https://github.com/user-attachments/assets/2d64f2fb-51f3-4805-8aae-5bd33d5f755f)

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

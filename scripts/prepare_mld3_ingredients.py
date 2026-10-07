"""Ingredient-level source masks for the observed strand branch."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from PIL import Image

SAM_ROOT = Path('/host/space0/guo-z/Evol-SAM3')


def build_processor():
    sys.path.insert(0, str(SAM_ROOT))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(bpe_path=str(SAM_ROOT / 'assets/bpe_simple_vocab_16e6.txt.gz'),
        checkpoint_path=str(SAM_ROOT / 'sam3/sam3.pt'), load_from_HF=False, device='cuda', eval_mode=True)
    return Sam3Processor(model, confidence_threshold=.30)


def source_noodle_mask(processor, image_state, food_mask):
    positives = ['noodles', 'pasta', 'noodle strands', 'soba noodles', 'udon noodles', 'ramen noodles', 'spaghetti']
    prompts = positives + ['ginger', 'bonito flakes', 'fish flakes', 'seaweed', 'vegetables', 'egg', 'meat', 'spoon', 'chopsticks']
    records, masks = [], []
    for prompt in prompts:
        processor.reset_all_prompts(image_state)
        result = processor.set_text_prompt(state=image_state, prompt=prompt)
        found = result['masks'].detach().cpu().numpy().astype(bool).reshape(-1, *food_mask.shape)
        scores = result['scores'].detach().cpu().numpy().reshape(-1)
        for mask, score in zip(found, scores):
            overlap = int((mask & food_mask).sum())
            records.append(dict(prompt=prompt, score=float(score), pixels=int(mask.sum()),
                source_food_overlap=overlap, inside_source_food=float(overlap / max(1, int(mask.sum())))))
            masks.append(mask)
    positive = np.zeros(food_mask.shape, bool)
    for record, mask in zip(records, masks):
        if record['prompt'] in positives and record['score'] >= .35 and record['inside_source_food'] >= .50:
            positive |= mask & food_mask
    excluded = np.zeros_like(positive)
    for record, mask in zip(records, masks):
        fraction = (mask & positive).sum() / max(1, int(positive.sum()))
        if record['prompt'] not in positives and record['score'] >= .35 and .001 < fraction < .65:
            excluded |= mask & positive
    ingredient = positive & ~excluded
    metadata = dict(rule='Source SAM3 noodle noun-bank masks intersect source food, minus confident source garnish/other-ingredient masks covering less than 65% of noodle support.',
        prompts=prompts, detections=records, noodle_support_pixels=int(positive.sum()),
        excluded_pixels=int(excluded.sum()), ingredient_pixels=int(ingredient.sum()),
        target_image_used=False, semantic_identity='SAM3 source ingredient hypothesis, independently visually audited.')
    return ingredient, excluded, np.asarray(masks, bool), metadata


def prepare_folder(folder, processor):
    image = Image.open(folder / 'source.png').convert('RGB')
    food = np.asarray(Image.open(folder / 'food_mask.png')) > 0
    image_state = processor.set_image(image)
    processor.reset_all_prompts(image_state)
    result = processor.set_text_prompt(state=image_state, prompt='food')
    dish_masks = result['masks'].detach().cpu().numpy().astype(bool).reshape(-1, *food.shape)
    dish_scores = result['scores'].detach().cpu().numpy().reshape(-1)
    dish = food.copy()
    for mask, score in zip(dish_masks, dish_scores):
        if score >= .50 and .006 < mask.mean() < .88:
            dish |= mask
    ingredient, excluded, masks, metadata = source_noodle_mask(processor, image_state, dish)
    metadata['original_selected_food_pixels'] = int(food.sum())
    metadata['source_food_union_pixels'] = int(dish.sum())
    if not (folder / 'selected_food_mask.png').exists():
        Image.fromarray(food.astype(np.uint8) * 255).save(folder / 'selected_food_mask.png')
    Image.fromarray(dish.astype(np.uint8) * 255).save(folder / 'food_mask.png')
    Image.fromarray(ingredient.astype(np.uint8) * 255).save(folder / 'noodle_mask.png')
    Image.fromarray(excluded.astype(np.uint8) * 255).save(folder / 'excluded_ingredients.png')
    np.savez_compressed(folder / 'ingredient_candidates.npz', masks=masks)
    (folder / 'ingredients.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print('source_ingredients', folder.name, metadata['ingredient_pixels'], metadata['excluded_pixels'], flush=True)
    return metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    processor = build_processor()
    for folder in sorted(args.root.glob('real_*')):
        prompt = json.loads((folder / 'segmentation.json').read_text())['selected_prompt'].lower()
        if any(x in prompt for x in ['noodle', 'spaghetti', 'udon', 'ramen']):
            prepare_folder(folder, processor)


if __name__ == '__main__':
    main()

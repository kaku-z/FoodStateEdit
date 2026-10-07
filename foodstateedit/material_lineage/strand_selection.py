"""Choose a visible first bite from source ingredient segmentation and curves."""
import numpy as np
from .strands import reconstruct_strands


def select_visible_strand_bite(source, ingredient_mask, points, axes=None, *,
                              old_radius=None, minimum_span_pixels=24., max_curves=96):
    mask = np.asarray(ingredient_mask,bool) & np.isfinite(points).all(axis=-1)
    observed = reconstruct_strands(source,mask,points,None,axes,max_curves=max_curves)
    if not observed.curves:
        return np.zeros(mask.shape,bool),None,0.,dict(status='no_observed_strands',curve_count=0,
            sufficient_visible_bite=False,ingredient_segmentation_required=True)
    coordinates=np.concatenate([c['source_uv'] for c in observed.curves])
    lengths=np.concatenate([np.full(len(c['source_uv']),c['source_path_length_pixels']/len(c['source_uv'])) for c in observed.curves])
    coherence=np.concatenate([np.full(len(c['source_uv']),min(4.,c['source_path_length_pixels']/minimum_span_pixels)) for c in observed.curves])
    candidates=np.concatenate([c['source_uv'][np.unique(np.rint(np.linspace(0,len(c['source_uv'])-1,7)).astype(int))] for c in observed.curves])
    minimum_radius=max(float(old_radius or 0),.085*min(mask.shape),.60*minimum_span_pixels)
    maximum_radius=max(minimum_radius,.22*min(mask.shape))
    radii=np.unique(np.minimum(minimum_radius*np.array([1.,1.3,1.6,2.]),maximum_radius))
    scores=[]
    for radius in radii:
        for center in candidates:
            chosen=np.linalg.norm(coordinates-center,axis=1)<radius
            if chosen.sum()<3:continue
            span=float(np.max(np.ptp(coordinates[chosen],axis=0)))
            observed_length=float(lengths[chosen].sum())
            enough=span>=minimum_span_pixels and observed_length>=1.5*minimum_span_pixels
            density=float((lengths[chosen]*coherence[chosen]).sum()/radius)
            # Adequate observed extent precedes density; larger source regions
            # are considered explicitly rather than enlarging generated food.
            scores.append((enough,density,span,observed_length,float(radius),center.copy()))
    adequate=[s for s in scores if s[0]]
    if adequate:
        smallest=min(s[4] for s in adequate)
        enough,density,span,observed_length,radius,center=max([s for s in adequate if s[4]==smallest],key=lambda s:(s[1],s[2]))
    else:
        enough,density,span,observed_length,radius,center=max(scores,key=lambda s:(s[2],s[3],s[1]))
    yy,xx=np.indices(mask.shape);patch=((xx-center[0])**2+(yy-center[1])**2<radius**2)&mask
    metrics=dict(status='selected' if enough else 'insufficient_visible_strands',sufficient_visible_bite=bool(enough),
        candidate_count=len(scores),all_source_visible_curves=len(observed.curves),source_graph_density_score=density,
        selected_source_span_pixels=span,selected_observed_curve_length_pixels=observed_length,
        minimum_span_pixels=float(minimum_span_pixels),radius_pixels=radius,selected_source_pixels=int(patch.sum()),
        rule='Ingredient-level source mask; smallest source radius meeting visible span/length, then highest long-path density at that radius. Source region grows only if inadequate; target food is never scaled.',
        semantic_scope='SAM ingredient mask is estimated identity; observed path connectivity and occluded spans remain uncertain.',
        generated_shape_used=False,target_reference_used=False)
    return patch,center,radius,metrics

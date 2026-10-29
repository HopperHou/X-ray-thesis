"""Explicit process-local geometric extension. Frozen source files stay unchanged."""
import math
import torch
import v6a2_loss as original
from e2_geometry import centre_components

BASE_AXIS=original.axis_center_term
BASE_REDUCTION=original.assigned_geometry_loss
ARM='AXIS'
SCALE=1.0
CAPTURE=False
TENSORS={}

def axis_hook(pred, gt, stride):
    axis=BASE_AXIS(pred,gt,stride)
    _,iso=centre_components(pred,gt,stride)
    if CAPTURE:TENSORS.update(axis=axis,isotropic=iso)
    return iso*SCALE if ARM=='O_ISOTROPIC' else axis

def reduction_hook(iou,d2,pixel_boxes,variant,axis_center=None):
    if CAPTURE and len(iou):TENSORS['overlap']=2.0-2.0**iou
    if ARM=='O_ONLY' and variant=='axis_center_v1':
        if iou.numel()==0:return pixel_boxes.sum()*0.0
        return (2.0-2.0**iou).mean()
    return BASE_REDUCTION(iou,d2,pixel_boxes,variant,axis_center)

def install(arm='AXIS',scale=1.,capture=False):
    global ARM,SCALE,CAPTURE,TENSORS
    assert arm in ('B2','AXIS','O_ONLY','O_ISOTROPIC')
    assert math.isfinite(scale) and scale>0
    ARM,SCALE,CAPTURE,TENSORS=arm,scale,capture,{}
    original.axis_center_term=axis_hook
    original.assigned_geometry_loss=reduction_hook

def uninstall():
    original.axis_center_term=BASE_AXIS
    original.assigned_geometry_loss=BASE_REDUCTION

class AuditProxy:
    def __init__(self,criterion):object.__setattr__(self,'criterion',criterion)
    def __getattr__(self,k):return getattr(self.criterion,k)
    def __setattr__(self,k,v):setattr(self.criterion,k,v)
    def __call__(self,*args,**kw):
        loss,audit=self.criterion(*args,**kw)
        stats=audit.get('geo_stats',{})
        if ARM=='O_ISOTROPIC' and 'axis_center' in stats:
            stats['scaled_isotropic_center']=stats.pop('axis_center')
        if ARM=='O_ONLY' and 'axis_center' in stats:
            stats['unused_axis_diagnostic']=stats.pop('axis_center')
        audit['extension_arm']=ARM;audit['isotropic_scale']=SCALE if ARM=='O_ISOTROPIC' else None
        return loss,audit

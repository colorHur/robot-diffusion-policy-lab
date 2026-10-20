"""Bind the intended EGL context before robosuite renders/reads a framebuffer.

Some hard resets / context destruction leave no current GL context. This hook
only binds GL state; it does not change simulation state, policy, or RNG.
Upstream sources remain unmodified.
"""
def install():
    from robosuite.utils.binding_utils import MjRenderContext
    if getattr(MjRenderContext,'_franka_lab_context_binding',False):return
    for name in ('render','read_pixels'):
        original=getattr(MjRenderContext,name)
        def bound(self,*args,_original=original,**kwargs):
            self.gl_ctx.make_current()
            return _original(self,*args,**kwargs)
        setattr(MjRenderContext,name,bound)
    MjRenderContext._franka_lab_context_binding=True

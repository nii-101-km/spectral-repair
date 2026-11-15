/* Mutable source/mask/view state is explicit and private to each editor instance. */
function createEditorState() {
  return {source:null, grid:null, images:[], plots:[], view:null, tool:'rectangle',
    overlay:null, overlayImage:null, dirty:true, frame:0, drag:null, cursor:null};
}

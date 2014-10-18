
var Editor = function(i) {
	var e = $('#editor_main')[0];
	e.width = i.width;
	e.height = i.height;

	// Basic, immutable settings.
	this.ctx = e.getContext('2d');
	this.width = i.width;
	this.height = i.height;
	this.image = i;

	// Variables that editor modifies.
	this.model = [];
};

Editor.prototype.redraw = function() {
	var ctx = this.ctx;
	ctx.drawImage(this.image, 0, 0);

	// update info
	$('#editor_info').text(JSON.stringify(this.model, null, 2));

	// draw grid region
	ctx.beginPath();
	ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
	ctx.lineWidth = 2;
	_.each(this.model["corners"], function(corner, i) {
		ctx.lineTo(corner[0], corner[1]);
	});
	ctx.closePath();
	ctx.stroke();
};


var curr_ix = 0;
var editor = null;

function recreate_editor_from_current_index() {
	editor = new Editor($('img')[curr_ix]);
	editor.model = metadata['' + curr_ix];
	editor.redraw();
}

$('#btn_prev').click(function() {
	curr_ix = Math.max(0, curr_ix - 1);
	recreate_editor_from_current_index();
});

$('#btn_next').click(function() {
	curr_ix = Math.min($('img').length - 1, curr_ix + 1);
	recreate_editor_from_current_index();
});

recreate_editor_from_current_index();

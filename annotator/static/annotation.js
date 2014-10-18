
var Editor = function(i) {
	var std_width = 600;

	var e = $('#editor_main')[0];

	// Basic, immutable settings.
	this.img_width = i.width;
	this.img_height = i.height;
	this.editor_width = std_width;
	this.editor_height = Math.floor(i.height * this.editor_width / i.width);
	this.image = i;

	// Variables that editor modifies.
	this.model = [];

	// Set UI size.
	e.width = this.editor_width;
	e.height = this.editor_height;
	this.ctx = e.getContext('2d');
};

Editor.prototype.redraw = function() {
	var ctx = this.ctx;
	ctx.save();
	ctx.scale(
		this.editor_width / this.img_width,
		this.editor_height / this.img_height);
	// Operate in image-space hereafter.

	ctx.drawImage(this.image, 0, 0);

	// update info
	$('#editor_info').text(JSON.stringify(this.model, null, 2));

	// draw grid region
	ctx.beginPath();
	ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
	ctx.lineWidth = 3;
	_.each(this.model["corners"], function(corner) {
		ctx.lineTo(corner[0], corner[1]);
	});
	ctx.closePath();
	ctx.stroke();
	if(this.model["corners"].length > 0) {
		var pt = this.model["corners"][0];
		ctx.beginPath();
		ctx.arc(pt[0], pt[1], 5, 0, 2 * Math.PI);
		ctx.stroke();
	}

	ctx.restore();
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

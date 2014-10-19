
var Editor = function(i) {
	var std_width = 700;

	var e = $('#the_editor > .editor_main')[0];

	// Basic, immutable settings.
	this.img_width = i.width;
	this.img_height = i.height;
	this.editor_width = std_width;
	this.editor_height = Math.floor(i.height * this.editor_width / i.width);
	this.image = i;

	// temporary editor state for UI
	this.corners_in_edit = [];

	// Variables that editor modifies.
	this.model = [];

	var _this = this;
	$('#the_editor > .editor_main').unbind('click');
	$('#the_editor > .editor_main').click(function(ev) {
		_this.corners_in_edit.push(
			_this.screen_to_image(
				[ev.offsetX, ev.offsetY]));
		// Write to model when completed.
		if(_this.corners_in_edit.length == 4) {
			_this.model["corners"] = _this.corners_in_edit;
			_this.corners_in_edit = [];
			_this.sync_corners();
		}
		_this.redraw();
	});

	$('#the_editor > .editor_lgtm').unbind('click');
	$('#the_editor > .editor_lgtm').click(function() {
		_this.sync_corners();
		_this.redraw();
	});

	$('#the_editor > .editor_sync_config').unbind('click');
	$('#the_editor > .editor_sync_config').click(function() {
		_this.sync_config();
		_this.redraw();
	});

	// Set UI size.
	e.width = this.editor_width;
	e.height = this.editor_height;
	this.ctx = e.getContext('2d');
};

Editor.prototype.sync_corners = function() {
	// Send this.model["corners"] to server.
	// (with groudtruth flag=true)
	$.ajax({
		type: 'post',
		url: '/photo/' + this.model.id,
		data: JSON.stringify({
			"corners": this.model["corners"],
		}),
		dataType: 'json',
		contentType: 'application/json'
	});
};

Editor.prototype.sync_config = function() {
	$.ajax({
		type: 'post',
		url: '/photo/' + this.model.id,
		data: JSON.stringify({
			"config": this.model["config"]
		}),
		dataType: 'json',
		contentType: 'application/json'
	});
};

Editor.prototype.screen_to_image = function(pt) {
	var scale_s_to_i = this.img_width / this.editor_width;
	return [pt[0] * scale_s_to_i, pt[1] * scale_s_to_i];
};

Editor.prototype.redraw = function() {
	// Canvas.
	this.ctx.save();
	this.ctx.scale(
		this.editor_width / this.img_width,
		this.editor_height / this.img_height);
	try {
		this._redraw_canvas_image_space(this.ctx);
	} catch(e) {
		console.log(e);
	}
	this.ctx.restore();

	// update info
	var info = _.omit(this.model, 'image_uri_encoded', 'corners', 'config');
	$('#the_editor > .editor_info').text(JSON.stringify(info, null, 2));

	// update table.
	var cell_tables = $('#the_editor > .editor_config');
	cell_tables.empty();
	// header
	var h_row = $('<tr/>');
	_.each(_.range(9, 0, -1), function(x) {
		h_row.append($('<th/>').text(x.toString()));
	});
	cell_tables.append(h_row);
	// body
	var _this = this;
	_.each(_.range(1, 10), function(y) {
		var row = $('<tr/>');
		_.each(_.range(9, 0, -1), function(x) {
			var key = x.toString() + y.toString();
			var val = _this.model["config"][key];

			var type_edit = $('<input/>').attr('size', 5).val('empty');
			var state_edit = $('<select/>');
			state_edit.append($('<option/>').val('up').text('↑'));
			state_edit.append($('<option/>').val('down').text('↓'));
			state_edit.append($('<option/>').val('empty').text('-').attr('selected', 'selected'));

			state_edit.change(function(ev) {
				if(ev.target.value === 'empty') {
					type_edit.val('empty');
				}
			});

			if(val !== undefined) {
				type_edit.val(val.type);
				state_edit.val(val.state);
			} else {
				_this.model["config"][key] = {
					state: "empty",
					type: "empty"
				};
			}

			var cell = $('<td/>');
			cell.append(type_edit);
			cell.append($('<br/>'));
			cell.append(state_edit);
			row.append(cell);
		});
		cell_tables.append(row);
	});
};

Editor.prototype._redraw_canvas_image_space = function(ctx) {
	ctx.drawImage(this.image, 0, 0);

	// existing quad.
	this._drawQuad(ctx, this.model["corners"], "rgba(255, 0, 0, 0.5)");

	// modifying quad.
	this._drawQuad(ctx, this.corners_in_edit, "rgba(0, 0, 255, 0.5)");
};

Editor.prototype._drawQuad = function(ctx, corners, color) {
	ctx.beginPath();
	ctx.strokeStyle = color;
	ctx.lineWidth = 3;
	_.each(corners, function(corner) {
		ctx.lineTo(corner[0], corner[1]);
	});
	if(corners.length == 4) {
		ctx.closePath();
	}
	ctx.stroke();
	if(corners.length > 0) {
		var pt = corners[0];
		ctx.beginPath();
		ctx.arc(pt[0], pt[1], 5, 0, 2 * Math.PI);
		ctx.stroke();
	}
};


$.ajax('/photos').done(function(resp) {
	var metadata = resp.results;
	var curr_ix = 0;
	var editor = null;

	function update_nav_status() {
		$('#nav_status').text((curr_ix + 1) + ' / ' + metadata.length);
	}

	function recreate_editor_from_current_index() {
		$.ajax('/photo/' + metadata[curr_ix].id).done(function(resp) {
			var img = new Image();
			img.src = resp.image_uri_encoded;
			editor = new Editor(img);
			editor.model = resp;
			editor.redraw();
		});
	}

	$('#btn_prev').click(function() {
		curr_ix = Math.max(0, curr_ix - 1);
		update_nav_status();
		recreate_editor_from_current_index();
	});

	$('#btn_next').click(function() {
		curr_ix = Math.min(metadata.length - 1, curr_ix + 1);
		update_nav_status();
		recreate_editor_from_current_index();
	});

	$('#btn_all').click(function() {
		$.ajax('/photos').done(function(resp) {
			metadata = resp.results;
			curr_ix = 0;
			update_nav_status();
			recreate_editor_from_current_index();
		});
	});

	$('#btn_uncertain_corners').click(function() {
		$.ajax('/photos?uncertain_corners').done(function(resp) {
			metadata = resp.results;
			curr_ix = 0;
			update_nav_status();
			recreate_editor_from_current_index();
		});
	});

	update_nav_status();
	recreate_editor_from_current_index();
});

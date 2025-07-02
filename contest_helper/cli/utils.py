from importlib.resources import files


def load_template(template_name):
    template_path = files("contest_helper.cli.templates").joinpath(template_name)
    return template_path.read_text(encoding="utf-8")

"""Prompt 模板加载与渲染。"""
import os
from jinja2 import Environment, FileSystemLoader

_env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))


def render_prompt(template_name, **kwargs):
    """加载并渲染指定模板，返回最终的 prompt 字符串。"""
    template = _env.get_template(template_name)
    return template.render(**kwargs)

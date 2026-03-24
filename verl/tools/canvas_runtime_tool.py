"""Canvas runtime helpers for RL-side agentic interaction."""

import os
import warnings
from typing import Any, Optional

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from playwright.sync_api import sync_playwright


warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


class CanvasNotebookState:
    """HTML notebook state for RL-side Canvas interaction."""

    def __init__(self, initial_svg: Optional[str] = None):
        try:
            BeautifulSoup("<b></b>", "lxml")
            self.parser = "lxml"
        except Exception:
            self.parser = "html.parser"

        self.state = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NoteBook State</title>
    <style>
    *,
    *::before,
    *::after {
      box-sizing: border-box;
    }

    body {
      font-family: sans-serif;
      line-height: 1.6;
      background-color: #f4f7f9;
      color: #333;
      margin: 0;
      padding: 20px;
    }

    main {
      max-width: 800px;
      margin: 20px auto;
      padding: 20px;
      background-color: #ffffff;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }

    h1, h2 {
      border-bottom: 2px solid #e0e0e0;
      padding-bottom: 10px;
      color: #1a2c3b;
    }

    p {
      margin-top: 0;
    }

    section {
      margin-bottom: 30px;
    }

    section:nth-of-type(1) > div {
      padding: 20px;
      border: 2px dashed #666;
      background-color: #f0f0f0;
    }

    section:nth-of-type(1) > div > div {
      padding: 15px;
      border: 1px solid;
      color: #fff;
      margin-bottom: 10px;
    }

    section:nth-of-type(1) > div > div:first-of-type {
      background-color: #009E5F;
      border-color: #007A4B;
    }

    section:nth-of-type(1) > div > div:last-of-type {
      background-color: #ED2633;
      border-color: #B81E27;
      width: 550px;
    }

    main img {
      max-width: 100%;
      height: auto;
      display: block;
    }

    main figure {
      border: 2px solid #1377EB;
      padding: 10px;
      margin: 0 0 20px 0;
    }

    section:nth-of-type(3) > div {
      display: flex;
      gap: 20px;
    }

    section:nth-of-type(3) > div > article {
      flex: 1;
      border: 1px solid #ccc;
      padding: 15px;
      border-radius: 5px;
      background: #fafafa;
      min-height: 120px;
    }

    section:nth-of-type(4) > div {
      border: 2px solid #f2994a;
      padding: 15px;
      background-color: #fff8f2;
      overflow-wrap: break-word;
      word-wrap: break-word;
    }

    section:nth-of-type(4) > div > p:last-of-type {
      font-size: 0.9em;
      color: #777;
    }
  </style>
</head>
<body>
  <div id="root"></div>
</body>
</html>
"""
        if initial_svg:
            try:
                soup_temp = BeautifulSoup(initial_svg, self.parser)
                tag = soup_temp.find()
                if tag and not tag.get("id"):
                    pass
            except Exception:
                pass
            self.update_state(action="insert_element", attrs={"fragment": initial_svg, "rootId": None})

    def _allow_external_images(self) -> bool:
        return os.environ.get("BLACKBOARD_ALLOW_EXTERNAL_IMAGES", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

    def _is_external_url(self, url: str) -> bool:
        u = (url or "").strip().lower()
        return u.startswith("http://") or u.startswith("https://")

    def _sanitize_external_images(self, fragment_soup: BeautifulSoup) -> None:
        """Remove or replace external image URLs to keep rendering offline-safe."""
        if self._allow_external_images():
            return

        for img in list(fragment_soup.find_all("img")):
            src = img.get("src")
            if isinstance(src, str) and self._is_external_url(src):
                replacement = fragment_soup.new_tag(
                    "div",
                    attrs={
                        "style": "color:#ED2633;font-size:14px;"
                        "padding:8px 0;"
                        "word-break:break-word;"
                    },
                )
                replacement.string = f"[blocked external image] {src}"
                img.replace_with(replacement)

        for svg_img in list(fragment_soup.find_all("image")):
            href = svg_img.get("href") or svg_img.get("xlink:href")
            if isinstance(href, str) and self._is_external_url(href):
                parent = svg_img.parent
                if parent and hasattr(parent, "new_tag"):
                    try:
                        x = svg_img.get("x") or "0"
                        y = svg_img.get("y") or "0"
                        text = fragment_soup.new_tag("text")
                        text["x"] = str(x)
                        text["y"] = str(y)
                        text["fill"] = "#ED2633"
                        text.string = "[blocked external svg image]"
                        svg_img.replace_with(text)
                    except Exception:
                        svg_img.decompose()
                else:
                    svg_img.decompose()

    def update_state(self, action: str, attrs: dict[str, Any]) -> None:
        """Update the notebook HTML state using one Canvas action."""
        soup = BeautifulSoup(self.state, self.parser)

        if action == "insert_element":
            fragment_str = attrs.get("fragment")
            if not fragment_str:
                print("警告 (insert): 'fragment' 不能为空。")
                return

            fragment_soup = BeautifulSoup(fragment_str, self.parser)
            self._sanitize_external_images(fragment_soup)

            new_element = fragment_soup.find()
            if not new_element:
                print(f"警告 (insert): 无法解析提供的fragment: {fragment_str}")
                return

            if not new_element.get("id"):
                new_element["id"] = "initial_svg" if new_element.name == "svg" else f"elem_{len(soup.find_all())}"

            root_id = attrs.get("rootId")
            parent_element = soup.find(id=root_id) if root_id else soup.body
            if not parent_element:
                if root_id == "root":
                    print("警告 (insert): 找不到ID为 'root' 的父节点，正在自动创建...")
                    new_root = soup.new_tag("div", id="root")
                    soup.body.append(new_root)
                    parent_element = new_root
                else:
                    print(f"警告 (insert): 找不到ID为 '{root_id}' 的父节点，将插入到<body>中。")
                    parent_element = soup.body

            before_id = attrs.get("beforeId")
            if before_id:
                before_element = soup.find(id=before_id)
                if before_element:
                    before_element.insert_before(new_element)
                else:
                    print(f"警告 (insert): 找不到ID为 '{before_id}' 的兄弟节点，将追加到末尾。")
                    parent_element.append(new_element)
            else:
                parent_element.append(new_element)

        elif action == "modify_element":
            target_id = attrs.get("targetId")
            attributes_to_update = attrs.get("attrs")
            if not target_id or not attributes_to_update:
                print("警告 (modify): 'targetId' 和 'attrs' 不能为空。")
                return

            target_element = soup.find(id=target_id)
            if target_element:
                for key, value in attributes_to_update.items():
                    if key == "text":
                        target_element.string = str(value)
                    else:
                        target_element[key] = str(value)
            else:
                print(f"警告 (modify): 找不到ID为 '{target_id}' 的元素。")

        elif action == "remove_element":
            target_id = attrs.get("targetId")
            if not target_id:
                print("警告 (remove): 'targetId' 不能为空。")
                return

            target_element = soup.find(id=target_id)
            if target_element:
                target_element.decompose()
            else:
                print(f"警告 (remove): 找不到ID为 '{target_id}' 的元素。")

        elif action == "clear_element" or action == "clear":
            if soup.body:
                soup.body.clear()
                new_root = soup.new_tag("div", id="root")
                soup.body.append(new_root)
            else:
                self.__init__()
                return

        elif action == "replace_element":
            target_id = attrs.get("targetId")
            fragment_str = attrs.get("fragment")
            if not target_id or not fragment_str:
                print("警告 (replace): 'targetId' 和 'fragment' 不能为空。")
                return

            target_element = soup.find(id=target_id)
            if target_element:
                new_element = BeautifulSoup(fragment_str, self.parser).find()
                if new_element:
                    target_element.replace_with(new_element)
                else:
                    print(f"警告 (replace): 无法解析提供的fragment: {fragment_str}")
            else:
                print(f"警告 (replace): 找不到ID为 '{target_id}' 的元素。")

        else:
            print(f"错误: 未知的操作 '{action}'")
            return

        self.state = str(soup)

    def render_state(self, output_path: str = "output.png") -> str:
        """Render the current notebook state to a high-resolution screenshot."""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            soup = BeautifulSoup(self.state, self.parser)

            if soup.body:
                children = [child for child in soup.body.contents if child.name]
                if len(children) > 3:
                    for child in children[:-3]:
                        child.decompose()

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 500, "height": 500}, device_scale_factor=2)
                page.set_content(str(soup))
                page.screenshot(path=output_path, full_page=True)
                browser.close()
                print(f"高清截图已保存: {output_path}")
                return "tool execute success"
        except Exception as e:
            print(f"tool execute failed: {e}")
            return f"tool execute failed: {e}"

    def insert_element(self, fragment: str, rootId: Optional[str] = None, beforeId: Optional[str] = None) -> None:
        """Wrapper for insert_element tool semantics."""
        self.update_state("insert_element", {"fragment": fragment, "rootId": rootId, "beforeId": beforeId})

    def modify_element(self, targetId: str, attrs: dict[str, Any]) -> None:
        """Wrapper for modify_element tool semantics."""
        self.update_state("modify_element", {"targetId": targetId, "attrs": attrs})

    def remove_element(self, targetId: str) -> None:
        """Wrapper for remove_element tool semantics."""
        self.update_state("remove_element", {"targetId": targetId})

    def replace_element(self, targetId: str, fragment: str) -> None:
        """Wrapper for replace_element tool semantics."""
        self.update_state("replace_element", {"targetId": targetId, "fragment": fragment})

    def clear(self) -> None:
        """Wrapper for clear tool semantics."""
        self.update_state("clear", {})
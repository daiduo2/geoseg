# pdf_extractor — 模块契约

> **M0.5-Fallback**：PyMuPDF 提取 + rasterize 补全。主提取是 `mineru_client/`。

## 职责

- 提取 PDF 内嵌的 `XObject(Image)` + 页面文字块
- `rasterize_page(pdf_path, page_idx, dpi=300)` 整页/区域 rasterize
- 当 MinerU 拆分 figure 或提取尺寸过小时作为 **fallback** 补全

## 与 mineru_client 的分工

| 场景 | 主责模块 | 行为 |
|------|----------|------|
| 标准提取 | `mineru_client` | MinerU v4 API → figure 图片 + caption markdown + content_list.json |
| 拆分补全 | `pdf_extractor` (fallback) | PyMuPDF rasterize 被拆分的 figure 区域 |
| 尺寸过小 | `pdf_extractor` (fallback) | PyMuPDF rasterize 原始 figure 区域为高分辨率 PNG |
| XObject 提取 | `pdf_extractor` | 提取 PDF 内嵌位图（概念模型图当前靠外部 `ph01.jpg` fallback） |

## 已知限制

当前概念模型图使用**位图 fallback**（`ph01.jpg`）：
- 矢量图形提取在并行 session 中开发，规格见 [`docs/PDF_VECTOR_EXTRACTION_SPEC.md`](../../../docs/PDF_VECTOR_EXTRACTION_SPEC.md)
- 矢量提取成熟后**替换数据源**，pipeline 内部接口不变

## 不做

- 不替代 `mineru_client` 做主提取（MinerU API 对 figure/caption 的提取能力更优）
- 不在本模块解析图内容（语义交给 `vlm_client`，几何交给 `cv_detect`）
- 不静默吞掉提取失败：缺嵌入图 → 抛错或落 `repair_hints`，由上层决定 fallback

## 测试

```bash
python -m geoseg.modules.pdf_extractor.demo
python -m geoseg.modules.pdf_extractor.demo_vector
```

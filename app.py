import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from rasterio.io import MemoryFile


st.set_page_config(
    page_title="Demo bản đồ nhiệt độ bề mặt Hà Nội",
    page_icon="🌡️",
    layout="wide",
)

st.title("🌡️ HỆ THỐNG DEMO BẢN ĐỒ NHIỆT ĐỘ BỀ MẶT (LST)")
st.caption("Dữ liệu Landsat 8/9 Collection 2 Level-2 · ST_B10 · QA_PIXEL · ST_QA tùy chọn · MTL")

with st.sidebar:
    st.header("Cấu hình xử lý")
    min_lst = st.number_input("Nhiệt độ thấp nhất (°C)", value=0.0, min_value=-50.0, max_value=80.0, step=1.0)
    max_lst = st.number_input("Nhiệt độ cao nhất (°C)", value=60.0, min_value=-20.0, max_value=100.0, step=1.0)
    use_stqa = st.checkbox("Sử dụng ST_QA để lọc độ không chắc chắn", value=False)
    max_stqa_k = st.number_input("Ngưỡng ST_QA (Kelvin)", value=5.0, min_value=0.1, max_value=20.0, step=0.5)
    display_size = st.select_slider("Kích thước cạnh lớn nhất khi hiển thị", options=[600, 800, 1000, 1200, 1400, 1600], value=1200)

st.subheader("1. Tải dữ liệu đầu vào")
c1, c2 = st.columns(2)
with c1:
    st_file = st.file_uploader("Ảnh ST_B10 (.TIF)", type=["tif", "tiff"], key="st")
    qa_file = st.file_uploader("Ảnh QA_PIXEL (.TIF)", type=["tif", "tiff"], key="qa")
with c2:
    mtl_file = st.file_uploader("Metadata MTL (.txt) – tùy chọn", type=["txt"], key="mtl")
    stqa_file = st.file_uploader("Ảnh ST_QA (.TIF) – tùy chọn", type=["tif", "tiff"], key="stqa")

st.subheader("Dữ liệu bổ sung cho phân tích nâng cao")
st.caption("Các mục dưới đây là tùy chọn. Chỉ tải các band thuộc cùng cảnh Landsat và cùng hệ quy chiếu.")
b1, b2, b3 = st.columns(3)
with b1:
    band3_file = st.file_uploader("Band 3 – Green (SR_B3)", type=["tif", "tiff"], key="sr_b3")
    band4_file = st.file_uploader("Band 4 – Red (SR_B4)", type=["tif", "tiff"], key="sr_b4")
with b2:
    band5_file = st.file_uploader("Band 5 – NIR (SR_B5)", type=["tif", "tiff"], key="sr_b5")
    band6_file = st.file_uploader("Band 6 – SWIR1 (SR_B6)", type=["tif", "tiff"], key="sr_b6")
with b3:
    usgs_file = st.file_uploader("LST tham chiếu USGS (GeoTIFF)", type=["tif", "tiff"], key="usgs")
    boundary_file = st.file_uploader("Ranh giới vùng (GeoJSON)", type=["geojson", "json"], key="boundary")

st.info(
    "Chọn ST_B10 và QA_PIXEL thuộc cùng một cảnh Landsat. "
    "Nếu bật ST_QA, hãy tải thêm đúng ảnh ST_QA cùng cảnh."
)

def parse_mtl(raw_bytes):
    if not raw_bytes:
        return {}
    text = raw_bytes.decode("utf-8", errors="ignore")
    wanted = [
        "UTM_ZONE",
        "CORNER_UL_PROJECTION_X_PRODUCT", "CORNER_UL_PROJECTION_Y_PRODUCT",
        "CORNER_UR_PROJECTION_X_PRODUCT", "CORNER_UR_PROJECTION_Y_PRODUCT",
        "CORNER_LL_PROJECTION_X_PRODUCT", "CORNER_LL_PROJECTION_Y_PRODUCT",
        "CORNER_LR_PROJECTION_X_PRODUCT", "CORNER_LR_PROJECTION_Y_PRODUCT",
    ]
    result = {}
    for key in wanted:
        match = re.search(r"(?m)^\s*" + re.escape(key) + r"\s*=\s*\"?([-+]?\d+(?:\.\d+)?)", text)
        if match:
            result[key] = float(match.group(1))
    return result


def open_tiff(raw_bytes):
    mem = MemoryFile(raw_bytes)
    ds = mem.open()
    return mem, ds


def process_scene(st_bytes, qa_bytes, mtl_bytes, stqa_bytes, min_c, max_c, use_qa, qa_limit, max_side):
    st_mem, st_ds = open_tiff(st_bytes)
    qa_mem, qa_ds = open_tiff(qa_bytes)
    stqa_mem = stqa_ds = None
    try:
        if st_ds.count < 1 or qa_ds.count < 1:
            raise ValueError("Không tìm thấy band raster hợp lệ.")
        if (st_ds.height, st_ds.width) != (qa_ds.height, qa_ds.width):
            raise ValueError("Kích thước QA_PIXEL không trùng với ST_B10.")
        if use_qa:
            if not stqa_bytes:
                raise ValueError("Bạn đã bật ST_QA nhưng chưa tải ảnh ST_QA.")
            stqa_mem, stqa_ds = open_tiff(stqa_bytes)
            if (st_ds.height, st_ds.width) != (stqa_ds.height, stqa_ds.width):
                raise ValueError("Kích thước ST_QA không trùng với ST_B10.")

        rows, cols = st_ds.height, st_ds.width
        total = rows * cols
        keys = ["fill", "dilated_cloud", "cirrus", "cloud", "cloud_shadow", "snow"]
        bit_counts = {k: 0 for k in keys}
        valid_dn_count = 0
        bad_qa_count = 0
        valid_qa_count = 0
        valid_combined_count = 0
        abnormal_count = 0
        valid_count = 0
        sum_temp = 0.0
        sum_sq = 0.0
        tmin = np.inf
        tmax = -np.inf
        hist_edges = np.linspace(min_c, max_c, 61)
        hist_counts = np.zeros(len(hist_edges) - 1, dtype=np.int64)

        # Đọc theo block để không nạp toàn bộ ảnh 59 triệu pixel vào RAM.
        for _, window in st_ds.block_windows(1):
            dn = st_ds.read(1, window=window).astype(np.float32)
            qa = qa_ds.read(1, window=window).astype(np.uint16)

            valid_dn = np.isfinite(dn) & (dn > 0) & (dn < 65535)
            flags = [
                (qa & (1 << 0)) != 0,
                (qa & (1 << 1)) != 0,
                (qa & (1 << 2)) != 0,
                (qa & (1 << 3)) != 0,
                (qa & (1 << 4)) != 0,
                (qa & (1 << 5)) != 0,
            ]
            for key, flag in zip(keys, flags):
                bit_counts[key] += int(np.count_nonzero(flag))

            bad = np.logical_or.reduce(flags)
            valid_qa = ~bad
            valid_dn_count += int(valid_dn.sum())
            bad_qa_count += int(bad.sum())
            valid_qa_count += int(valid_qa.sum())

            valid = valid_dn & valid_qa
            if use_qa and stqa_ds is not None:
                stqa = stqa_ds.read(1, window=window).astype(np.float32)
                valid_stqa = np.isfinite(stqa) & (stqa >= 0) & ((stqa * 0.01) <= qa_limit)
                valid &= valid_stqa

            lst = dn * 0.00341802 + 149.0 - 273.15
            lst[~valid] = np.nan
            valid_combined_count += int(valid.sum())

            in_range = valid & (lst >= min_c) & (lst <= max_c)
            abnormal_count += int((valid & ~((lst >= min_c) & (lst <= max_c))).sum())
            values = lst[in_range]
            if values.size:
                valid_count += int(values.size)
                sum_temp += float(values.sum(dtype=np.float64))
                sum_sq += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
                tmin = min(tmin, float(values.min()))
                tmax = max(tmax, float(values.max()))
                hist_counts += np.histogram(values, bins=hist_edges)[0]

        if valid_count == 0:
            raise ValueError("Không còn pixel hợp lệ. Hãy kiểm tra dữ liệu hoặc nới ngưỡng nhiệt độ.")

        mean = sum_temp / valid_count
        variance = max(0.0, sum_sq / valid_count - mean * mean)
        std = float(np.sqrt(variance))
        median_bin = int(np.searchsorted(np.cumsum(hist_counts), valid_count / 2, side="left"))
        median_bin = min(median_bin, len(hist_edges) - 2)
        median_approx = float((hist_edges[median_bin] + hist_edges[median_bin + 1]) / 2)

        # Tạo ảnh xem trước có giới hạn kích thước.
        scale = min(1.0, max_side / max(rows, cols))
        out_h = max(1, int(round(rows * scale)))
        out_w = max(1, int(round(cols * scale)))
        dn_small = st_ds.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest).astype(np.float32)
        qa_small = qa_ds.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest).astype(np.uint16)
        valid_small = np.isfinite(dn_small) & (dn_small > 0) & (dn_small < 65535)
        bad_small = np.zeros((out_h, out_w), dtype=bool)
        for bit in range(6):
            bad_small |= (qa_small & (1 << bit)) != 0
        valid_small &= ~bad_small

        if use_qa and stqa_ds is not None:
            stqa_small = stqa_ds.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest).astype(np.float32)
            valid_small &= np.isfinite(stqa_small) & (stqa_small >= 0) & ((stqa_small * 0.01) <= qa_limit)

        lst_small = dn_small * 0.00341802 + 149.0 - 273.15
        valid_temp_small = valid_small & (lst_small >= min_c) & (lst_small <= max_c)
        lst_small[~valid_temp_small] = np.nan
        bad_map = (~valid_temp_small).astype(np.uint8)

        # Ảnh ST_B10 grayscale stretch percentile.
        dn_valid_small = dn_small[valid_small]
        if dn_valid_small.size:
            lo, hi = np.percentile(dn_valid_small, [2, 98])
        else:
            lo, hi = 0, 1
        if hi <= lo:
            hi = lo + 1
        st_gray = np.clip((dn_small - lo) / (hi - lo), 0, 1)
        st_gray[~valid_small] = 0

        mtl = parse_mtl(mtl_bytes)
        utm_keys = [
            "CORNER_UL_PROJECTION_X_PRODUCT", "CORNER_UL_PROJECTION_Y_PRODUCT",
            "CORNER_UR_PROJECTION_X_PRODUCT", "CORNER_UR_PROJECTION_Y_PRODUCT",
            "CORNER_LL_PROJECTION_X_PRODUCT", "CORNER_LL_PROJECTION_Y_PRODUCT",
        ]
        has_utm = all(k in mtl for k in utm_keys)

        stats = {
            "Tổng số pixel": total,
            "Pixel DN hợp lệ": valid_dn_count,
            "Fill": bit_counts["fill"],
            "Dilated Cloud": bit_counts["dilated_cloud"],
            "Cirrus": bit_counts["cirrus"],
            "Cloud": bit_counts["cloud"],
            "Cloud Shadow": bit_counts["cloud_shadow"],
            "Snow": bit_counts["snow"],
            "Pixel QA bị loại": bad_qa_count,
            "Pixel QA hợp lệ": valid_qa_count,
            "Pixel hợp lệ sau kết hợp": valid_combined_count,
            "Pixel nhiệt độ ngoài ngưỡng": abnormal_count,
            "Pixel hợp lệ cuối cùng": valid_count,
            "LST nhỏ nhất (°C)": tmin,
            "LST lớn nhất (°C)": tmax,
            "LST trung bình (°C)": mean,
            "LST trung vị xấp xỉ (°C)": median_approx,
            "Độ lệch chuẩn (°C)": std,
        }
        csv_data = pd.DataFrame({"Chỉ tiêu": list(stats.keys()), "Giá trị": list(stats.values())}).to_csv(index=False).encode("utf-8-sig")

        # Sinh PNG ở bộ nhớ để người dùng tải xuống.
        def fig_png(fig):
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
            plt.close(fig)
            buffer.seek(0)
            return buffer.getvalue()

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(st_gray, cmap="gray", origin="upper")
        ax.set_title("Ảnh ST_B10 (ảnh xem trước)")
        ax.set_xlabel("Cột pixel")
        ax.set_ylabel("Hàng pixel")
        fig.colorbar(im, ax=ax, label="DN đã co giãn hiển thị")
        st_png = fig_png(fig)

        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(lst_small, cmap="turbo", vmin=min_c, vmax=max_c, origin="upper")
        ax.set_title("Bản đồ nhiệt độ bề mặt LST sau lọc QA")
        ax.set_xlabel("Cột pixel")
        ax.set_ylabel("Hàng pixel")
        fig.colorbar(im, ax=ax, label="Nhiệt độ (°C)")
        lst_png = fig_png(fig)

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.imshow(bad_map, cmap="gray", vmin=0, vmax=1, origin="upper")
        ax.set_title("Bản đồ pixel không hợp lệ / bị loại")
        ax.set_xlabel("Cột pixel")
        ax.set_ylabel("Hàng pixel")
        bad_png = fig_png(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        centers = (hist_edges[:-1] + hist_edges[1:]) / 2
        ax.bar(centers, hist_counts, width=np.diff(hist_edges) * 0.9, color="#4C9BD6", edgecolor="black")
        ax.set_title("Histogram phân bố nhiệt độ LST")
        ax.set_xlabel("Nhiệt độ (°C)")
        ax.set_ylabel("Số lượng pixel")
        ax.grid(alpha=0.25)
        hist_png = fig_png(fig)

        utm_png = None
        if has_utm:
            ulx = mtl["CORNER_UL_PROJECTION_X_PRODUCT"]
            uly = mtl["CORNER_UL_PROJECTION_Y_PRODUCT"]
            urx = mtl["CORNER_UR_PROJECTION_X_PRODUCT"]
            lly = mtl["CORNER_LL_PROJECTION_Y_PRODUCT"]
            fig, ax = plt.subplots(figsize=(9, 7))
            im = ax.imshow(lst_small, cmap="turbo", vmin=min_c, vmax=max_c,
                           extent=[ulx, urx, lly, uly], origin="upper", aspect="auto")
            ax.set_title("Bản đồ LST theo phạm vi tọa độ UTM trong MTL")
            ax.set_xlabel("Tọa độ UTM X (m)")
            ax.set_ylabel("Tọa độ UTM Y (m)")
            fig.colorbar(im, ax=ax, label="Nhiệt độ (°C)")
            zone = mtl.get("UTM_ZONE")
            if zone is not None:
                ax.text(0.02, 0.02, f"UTM Zone {zone:g}", transform=ax.transAxes,
                        color="white", bbox={"facecolor": "black", "alpha": 0.55})
            utm_png = fig_png(fig)

        # Keep the exact same filtered preview for all advanced products.
        # This avoids rebuilding the QA mask differently in the hotspot section.
        from affine import Affine
        preview_profile = st_ds.profile.copy()
        preview_profile.update(
            height=out_h, width=out_w, count=1, dtype="float32",
            nodata=-9999.0, compress="deflate",
            transform=st_ds.transform * Affine.scale(cols / out_w, rows / out_h),
        )

        return {
            "stats": stats,
            "csv": csv_data,
            "lst_preview": lst_small.copy(),
            "valid_preview": valid_temp_small.copy(),
            "preview_profile": preview_profile,
            "preview_crs": st_ds.crs,
            "st_png": st_png,
            "lst_png": lst_png,
            "bad_png": bad_png,
            "hist_png": hist_png,
            "utm_png": utm_png,
            "hist_edges": hist_edges,
            "hist_counts": hist_counts,
            "preview_shape": f"{out_h} × {out_w}",
            "original_shape": f"{rows} × {cols}",
        }
    finally:
        st_ds.close()
        qa_ds.close()
        st_mem.close()
        qa_mem.close()
        if stqa_ds is not None:
            stqa_ds.close()
        if stqa_mem is not None:
            stqa_mem.close()


run = st.button("🚀 BẮT ĐẦU XỬ LÝ VÀ TẠO BẢN ĐỒ", type="primary", use_container_width=True)
if run:
    if st_file is None or qa_file is None:
        st.error("Bạn cần tải lên cả ST_B10 và QA_PIXEL.")
    elif min_lst >= max_lst:
        st.error("Nhiệt độ thấp nhất phải nhỏ hơn nhiệt độ cao nhất.")
    else:
        try:
            with st.spinner("Đang đọc GeoTIFF, lọc QA và tính nhiệt độ LST. Ảnh lớn có thể mất vài phút..."):
                result = process_scene(
                    st_file.getvalue(),
                    qa_file.getvalue(),
                    mtl_file.getvalue() if mtl_file else None,
                    stqa_file.getvalue() if stqa_file else None,
                    float(min_lst), float(max_lst), bool(use_stqa),
                    float(max_stqa_k), int(display_size),
                )
            st.session_state["lst_demo_result"] = result
            st.success("Đã xử lý xong dữ liệu và tạo các sản phẩm bản đồ!")
        except Exception as exc:
            st.error(f"Không thể xử lý dữ liệu: {exc}")
            st.exception(exc)

result = st.session_state.get("lst_demo_result")
if result:
    st.divider()
    st.subheader("2. Kết quả thống kê")
    st.caption(f"Kích thước ảnh gốc: {result['original_shape']} · Kích thước ảnh xem trước: {result['preview_shape']}")
    s = result["stats"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("LST trung bình", f"{s['LST trung bình (°C)']:.2f} °C")
    k2.metric("LST nhỏ nhất", f"{s['LST nhỏ nhất (°C)']:.2f} °C")
    k3.metric("LST lớn nhất", f"{s['LST lớn nhất (°C)']:.2f} °C")
    k4.metric("Pixel hợp lệ", f"{s['Pixel hợp lệ cuối cùng']:,}")

    st.dataframe(pd.DataFrame({"Chỉ tiêu": list(s.keys()), "Giá trị": list(s.values())}), use_container_width=True, hide_index=True)

    st.subheader("3. Bản đồ và biểu đồ")
    tabs = st.tabs(["Bản đồ LST", "Ảnh ST_B10", "Pixel bị loại", "Histogram", "Bản đồ UTM"])
    with tabs[0]:
        st.image(result["lst_png"], use_container_width=True)
        st.download_button("Tải bản đồ LST PNG", result["lst_png"], "Ban_do_LST_HaNoi.png", "image/png")
    with tabs[1]:
        st.image(result["st_png"], use_container_width=True)
        st.download_button("Tải ảnh ST_B10 PNG", result["st_png"], "Anh_goc_ST_B10.png", "image/png")
    with tabs[2]:
        st.image(result["bad_png"], use_container_width=True)
        st.download_button("Tải bản đồ pixel bị loại PNG", result["bad_png"], "Ban_do_pixel_bi_loai.png", "image/png")
    with tabs[3]:
        st.image(result["hist_png"], use_container_width=True)
        st.download_button("Tải Histogram PNG", result["hist_png"], "Histogram_LST_HaNoi.png", "image/png")
    with tabs[4]:
        if result["utm_png"]:
            st.image(result["utm_png"], use_container_width=True)
            st.download_button("Tải bản đồ UTM PNG", result["utm_png"], "Ban_do_LST_UTM.png", "image/png")
        else:
            st.warning("Không tìm thấy đủ tọa độ góc trong MTL. Bản đồ UTM chưa được tạo; các bản đồ theo pixel vẫn có.")
    st.download_button("⬇️ Tải bảng thống kê CSV", result["csv"], "ThongKe_LST_HaNoi.csv", "text/csv")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Ban_do_LST_HaNoi.png", result["lst_png"])
        zf.writestr("Anh_goc_ST_B10.png", result["st_png"])
        zf.writestr("Ban_do_pixel_bi_loai.png", result["bad_png"])
        zf.writestr("Histogram_LST_HaNoi.png", result["hist_png"])
        if result["utm_png"]:
            zf.writestr("Ban_do_LST_UTM.png", result["utm_png"])
        zf.writestr("ThongKe_LST_HaNoi.csv", result["csv"])
    st.download_button(
        "📦 Tải toàn bộ kết quả (.ZIP)",
        zip_buffer.getvalue(),
        "KetQua_LST_HaNoi.zip",
        "application/zip",
        use_container_width=True,
    )

st.divider()
st.caption("Lưu ý: Ảnh hiển thị được giảm mẫu để tăng tốc; thống kê pixel được tính theo từng block trên dữ liệu raster gốc. Trung vị được ước lượng từ histogram.")

def _excel_bytes(frame):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="ThongKe")
    return output.getvalue()




# ---------------------------------------------------------------------
# PHAN TICH NANG CAO: chi so pho, doi chieu USGS, hotspot va ban giao
# ---------------------------------------------------------------------
def _read_preview(uploaded_file, out_shape, resampling=Resampling.bilinear):
    """Đọc raster ở độ phân giải xem trước để hạn chế RAM."""
    if uploaded_file is None:
        return None, None, None
    mem = MemoryFile(uploaded_file.getvalue())
    ds = mem.open()
    try:
        h, w = out_shape
        arr = ds.read(
            1,
            out_shape=(h, w),
            resampling=resampling,
            masked=True,
        ).astype(np.float32).filled(np.nan)
        # Cập nhật transform tương ứng với raster đã giảm mẫu.
        from affine import Affine
        transform = ds.transform * Affine.scale(ds.width / w, ds.height / h)
        profile = ds.profile.copy()
        profile.update(
            height=h, width=w, transform=transform, count=1,
            dtype="float32", nodata=-9999.0, compress="deflate"
        )
        return arr, profile, ds.crs
    finally:
        ds.close()
        mem.close()


def _png_bytes(array, title, cmap="turbo", vmin=None, vmax=None, label="Giá trị"):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("Cột pixel")
    ax.set_ylabel("Hàng pixel")
    fig.colorbar(im, ax=ax, label=label)
    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out.getvalue()


def _geotiff_bytes(array, profile):
    if profile is None:
        return None
    output = io.BytesIO()
    data = np.asarray(array, dtype=np.float32).copy()
    data[~np.isfinite(data)] = -9999.0
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(data, 1)
        output.write(mem.read())
    return output.getvalue()


def _landsat_sr(uploaded_file, shape):
    """Landsat Collection 2 Level-2 SR: reflectance = DN*0.0000275 - 0.2."""
    dn, profile, crs = _read_preview(uploaded_file, shape)
    if dn is None:
        return None, None, None
    sr = dn * 0.0000275 - 0.2
    sr[(dn <= 0) | ~np.isfinite(dn)] = np.nan
    return sr, profile, crs


def _safe_index(numerator, denominator):
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (numerator - denominator) / (numerator + denominator)
    value[~np.isfinite(value)] = np.nan
    return value


def _csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8-sig")



def _build_docx(stats_frame, title):
    from docx import Document
    from docx.shared import Pt
    doc = Document()
    doc.add_heading(title, 0)
    doc.add_paragraph("Báo cáo được tạo tự động từ ứng dụng Demo LST Hà Nội.")
    table = doc.add_table(rows=1, cols=2)
    table.style = "Light Shading Accent 1"
    hdr = table.rows[0].cells
    hdr[0].text = "Chỉ tiêu"
    hdr[1].text = "Giá trị"
    for _, row in stats_frame.iterrows():
        cells = table.add_row().cells
        cells[0].text = str(row["Chỉ tiêu"])
        val = row["Giá trị"]
        cells[1].text = f"{val:,.4f}" if isinstance(val, (float, np.floating)) else str(val)
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            run.font.name = "Arial"
            run.font.size = Pt(10)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


def _build_pdf(stats_frame, title):
    """PDF summary; reportlab is included in requirements."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import mm

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 8*mm),
             Paragraph("Báo cáo được tạo tự động từ ứng dụng Demo LST Hà Nội.", styles["BodyText"]),
             Spacer(1, 5*mm)]
    rows = [["Chỉ tiêu", "Giá trị"]]
    for _, row in stats_frame.iterrows():
        val = row["Giá trị"]
        if isinstance(val, (float, np.floating)):
            val = f"{val:,.4f}"
        else:
            val = str(val)
        rows.append([str(row["Chỉ tiêu"]), val])
    table = Table(rows, colWidths=[105*mm, 55*mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#173B57")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EEF3F7")]),
        ("PADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(table)
    doc.build(story)
    return buf.getvalue()


if result:
    st.divider()
    st.header("4. Phân tích nâng cao và sản phẩm bàn giao")

    # Read the same-scene reflectance bands at the existing preview resolution.
    try:
        target_shape = tuple(int(x.strip()) for x in result["preview_shape"].split("×"))
        advanced_products = {}
        sr_b3 = sr_b4 = sr_b5 = sr_b6 = None
        base_profile = None
        if any([band3_file, band4_file, band5_file, band6_file]):
            missing = [name for name, f in [
                ("SR_B3", band3_file), ("SR_B4", band4_file),
                ("SR_B5", band5_file), ("SR_B6", band6_file)
            ] if f is None]
            if missing:
                st.warning("Để tính đủ NDVI, NDBI và MNDWI, cần tải đủ SR_B3, SR_B4, SR_B5, SR_B6. Còn thiếu: " + ", ".join(missing))
            else:
                sr_b3, p3, crs3 = _landsat_sr(band3_file, target_shape)
                sr_b4, p4, crs4 = _landsat_sr(band4_file, target_shape)
                sr_b5, p5, crs5 = _landsat_sr(band5_file, target_shape)
                sr_b6, p6, crs6 = _landsat_sr(band6_file, target_shape)
                profiles = [p3, p4, p5, p6]
                crss = [crs3, crs4, crs5, crs6]
                if len({str(c) for c in crss}) > 1:
                    st.warning("Các band có CRS khác nhau. Hãy kiểm tra chúng thuộc cùng cảnh và cùng hệ tọa độ trước khi sử dụng.")
                ndvi = _safe_index(sr_b5, sr_b4)
                ndbi = _safe_index(sr_b6, sr_b5)
                mndwi = _safe_index(sr_b3, sr_b6)
                advanced_products.update({"NDVI": ndvi, "NDBI": ndbi, "MNDWI": mndwi})
                base_profile = p3
                st.subheader("4.1. Các chỉ số phổ")
                idx_tabs = st.tabs(["NDVI", "NDBI", "MNDWI"])
                for tab, name, arr, cmap in zip(
                    idx_tabs, ["NDVI", "NDBI", "MNDWI"],
                    [ndvi, ndbi, mndwi], ["RdYlGn", "RdYlBu_r", "BrBG"]
                ):
                    with tab:
                        png = _png_bytes(arr, f"Bản đồ {name}", cmap, -1, 1, name)
                        st.image(png, use_container_width=True)
                        st.download_button(f"Tải {name} PNG", png, f"Ban_do_{name}.png", "image/png", key=f"dl_{name}")
                        tiff = _geotiff_bytes(arr, base_profile)
                        st.download_button(f"Tải {name} GeoTIFF (ảnh xem trước)", tiff,
                                           f"Ban_do_{name}_preview.tif", "image/tiff", key=f"tif_{name}")
                index_rows = []
                for name, arr in advanced_products.items():
                    vals = arr[np.isfinite(arr)]
                    if vals.size:
                        index_rows.append({"Chỉ số": name, "Giá trị nhỏ nhất": float(vals.min()),
                                           "Giá trị lớn nhất": float(vals.max()),
                                           "Giá trị trung bình": float(vals.mean()),
                                           "Pixel hợp lệ": int(vals.size)})
                index_df = pd.DataFrame(index_rows)
                if not index_df.empty:
                    st.dataframe(index_df, use_container_width=True, hide_index=True)
                    st.download_button("Tải bảng chỉ số CSV", _csv_bytes(index_df),
                                       "ThongKe_NDVI_NDBI_MNDWI.csv", "text/csv")
        else:
            st.info("Tải đủ SR_B3, SR_B4, SR_B5, SR_B6 để mở chức năng NDVI/NDBI/MNDWI.")

        # Reuse the exact LST preview and QA mask produced by process_scene.
        # Re-reading ST_B10/QA_PIXEL here used to create a second, inconsistent mask
        # and could leave the hotspot image entirely blank.
        lst_preview = np.asarray(result.get("lst_preview"), dtype=np.float32).copy()
        valid = np.asarray(result.get("valid_preview"), dtype=bool).copy()
        st_profile = result.get("preview_profile")
        st_crs = result.get("preview_crs")
        if lst_preview.size and np.isfinite(lst_preview).any():
            st.subheader("4.2. Bản đồ điểm nóng nhiệt (Hotspot)")
            hotspot_pct = st.slider("Ngưỡng xác định vùng nóng – phân vị (%)", 70, 99, 90, 1)
            vals = lst_preview[np.isfinite(lst_preview)]
            if vals.size:
                threshold = float(np.percentile(vals, hotspot_pct))
                hotspot = valid & np.isfinite(lst_preview) & (lst_preview >= threshold)
                hotspot_display = np.where(valid, hotspot.astype(np.float32), np.nan)
                hot_png = _png_bytes(hotspot_display,
                                     f"Hotspot LST – ngưỡng P{hotspot_pct} ({threshold:.2f} °C)",
                                     "hot", 0, 1, "Vùng nóng (1 = nóng)")
                st.image(hot_png, use_container_width=True)
                hot_count = int(hotspot.sum())
                hot_area_pct = 100.0 * hot_count / max(1, int(np.isfinite(lst_preview).sum()))
                st.metric("Tỷ lệ pixel thuộc vùng nóng trên ảnh xem trước", f"{hot_area_pct:.2f}%")
                st.download_button("Tải bản đồ Hotspot PNG", hot_png, "Ban_do_Hotspot.png", "image/png")
                hot_tif = _geotiff_bytes(np.where(valid, hotspot.astype(np.float32), np.nan), st_profile)
                st.download_button("Tải Hotspot GeoTIFF (ảnh xem trước)", hot_tif,
                                   "Raster_Hotspot_preview.tif", "image/tiff")

            st.subheader("4.3. Đối chiếu với raster LST tham chiếu USGS")
            if usgs_file is not None:
                ref_arr, ref_profile, ref_crs = _read_preview(usgs_file, target_shape, Resampling.bilinear)
                unit = st.radio("Đơn vị raster USGS", ["Độ C", "Kelvin"], horizontal=True)
                if unit == "Kelvin":
                    ref_arr = ref_arr - 273.15
                # Require matching CRS and transform for a valid pixel-by-pixel comparison.
                same_crs = str(ref_crs) == str(st_crs)
                same_transform = (ref_profile is not None and st_profile is not None and
                                  np.allclose(tuple(ref_profile["transform"]), tuple(st_profile["transform"])))
                if not same_crs or not same_transform:
                    st.error("Raster USGS chưa cùng CRS/phạm vi lưới với ST_B10. Không thực hiện so sánh pixel để tránh sai lệch. Hãy đồng bộ raster trong QGIS/GDAL trước.")
                else:
                    mask = np.isfinite(lst_preview) & np.isfinite(ref_arr)
                    diff = np.full(target_shape, np.nan, dtype=np.float32)
                    diff[mask] = lst_preview[mask] - ref_arr[mask]
                    if mask.sum() > 0:
                        mae = float(np.mean(np.abs(diff[mask])))
                        rmse = float(np.sqrt(np.mean(np.square(diff[mask]))))
                        bias = float(np.mean(diff[mask]))
                        corr = float(np.corrcoef(lst_preview[mask], ref_arr[mask])[0, 1]) if mask.sum() > 2 else np.nan
                        compare_df = pd.DataFrame([
                            ["Số pixel so sánh", int(mask.sum())],
                            ["MAE (°C)", mae], ["RMSE (°C)", rmse],
                            ["Bias (°C)", bias], ["Pearson r", corr]
                        ], columns=["Chỉ tiêu", "Giá trị"])
                        st.dataframe(compare_df, use_container_width=True, hide_index=True)
                        err_png = _png_bytes(diff, "Raster sai số LST – Demo trừ USGS", "coolwarm",
                                             -max(abs(np.nanpercentile(diff[mask], 2)), abs(np.nanpercentile(diff[mask], 98)), 0.1),
                                             max(abs(np.nanpercentile(diff[mask], 2)), abs(np.nanpercentile(diff[mask], 98)), 0.1),
                                             "Sai số (°C)")
                        st.image(err_png, use_container_width=True)
                        st.download_button("Tải raster sai số PNG", err_png, "Raster_sai_so_LST.png", "image/png")
                        err_tif = _geotiff_bytes(diff, st_profile)
                        st.download_button("Tải raster sai số GeoTIFF (ảnh xem trước)", err_tif,
                                           "Raster_sai_so_LST_preview.tif", "image/tiff")
                        st.download_button("Tải bảng sai số CSV", _csv_bytes(compare_df),
                                           "Bang_sai_so_USGS.csv", "text/csv")
                    else:
                        st.warning("Không có pixel hợp lệ chung để so sánh.")
            else:
                st.info("Tải raster LST tham chiếu USGS cùng CRS và cùng lưới với ST_B10 để tính MAE, RMSE, Bias và tương quan.")

            st.subheader("4.4. Xếp hạng nhiệt độ theo vùng")
            st.caption("Bản hiện tại xếp hạng theo các ô lưới của ảnh xem trước; muốn xếp hạng theo quận/huyện cần tải ranh giới hành chính đã chuẩn hóa và bổ sung phép thống kê zonal.")
            block_rows = st.number_input("Số hàng chia lưới", min_value=2, max_value=20, value=4, step=1, key="rank_rows")
            block_cols = st.number_input("Số cột chia lưới", min_value=2, max_value=20, value=4, step=1, key="rank_cols")
            ranking = []
            hh, ww = lst_preview.shape
            for rr in range(int(block_rows)):
                for cc in range(int(block_cols)):
                    y0, y1 = rr * hh // int(block_rows), (rr + 1) * hh // int(block_rows)
                    x0, x1 = cc * ww // int(block_cols), (cc + 1) * ww // int(block_cols)
                    part = lst_preview[y0:y1, x0:x1]
                    part = part[np.isfinite(part)]
                    if part.size:
                        ranking.append({"Ô lưới": f"R{rr+1}-C{cc+1}", "LST trung bình (°C)": float(part.mean()),
                                        "LST lớn nhất (°C)": float(part.max()), "Số pixel": int(part.size)})
            rank_df = pd.DataFrame(ranking).sort_values("LST trung bình (°C)", ascending=False)
            rank_df.insert(0, "Thứ hạng", range(1, len(rank_df) + 1))
            st.dataframe(rank_df, use_container_width=True, hide_index=True)
            st.download_button("Tải bảng xếp hạng CSV", _csv_bytes(rank_df),
                               "Bang_xep_hang_nhiet_do_theo_o_luoi.csv", "text/csv")
            st.download_button("Tải bảng xếp hạng Excel", _excel_bytes(rank_df),
                               "Bang_xep_hang_nhiet_do.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            # GeoTIFF preview of LST; explicitly named preview to avoid claiming full-resolution output.
            st.subheader("4.5. Xuất raster GeoTIFF")
            lst_tif = _geotiff_bytes(lst_preview, st_profile)
            st.download_button("Tải LST GeoTIFF (ảnh xem trước giảm mẫu)",
                               lst_tif, "LST_HaNoi_preview.tif", "image/tiff")
        else:
            st.warning("Không có pixel LST hợp lệ để tạo Hotspot/xếp hạng. Hãy kiểm tra ST_B10, QA_PIXEL và nới khoảng nhiệt độ ở thanh bên.")

        # Optional boundary validation/display (GeoJSON only; no fabricated geometry).
        if boundary_file is not None:
            st.subheader("4.6. Ranh giới vùng nghiên cứu")
            try:
                import json
                geo = json.loads(boundary_file.getvalue().decode("utf-8-sig"))
                features = geo.get("features", []) if isinstance(geo, dict) else []
                rows = []
                for index, feature in enumerate(features, start=1):
                    props = feature.get("properties") or {}
                    rows.append({"STT": index, **props})
                st.write(f"Số đối tượng ranh giới: {len(features)}")
                if rows:
                    st.dataframe(pd.DataFrame(rows).head(100), use_container_width=True, hide_index=True)
                else:
                    st.warning("GeoJSON không có đối tượng trong mục 'features'.")
                st.caption("Đã đọc GeoJSON bằng bộ đọc tích hợp. Để thống kê chính xác theo quận/huyện, cần kiểm tra CRS và thực hiện phép thống kê vùng (zonal statistics).")
            except Exception as exc:
                st.error(f"Không đọc được GeoJSON: {exc}")

    except Exception as exc:
        st.error(f"Lỗi ở nhóm phân tích nâng cao: {exc}")
        st.exception(exc)

    st.subheader("5. Xuất báo cáo và gói bàn giao")
    summary_frame = pd.DataFrame({"Chỉ tiêu": list(result["stats"].keys()),
                                  "Giá trị": list(result["stats"].values())})
    r1, r2 = st.columns(2)
    with r1:
        st.download_button("Tải báo cáo thống kê Excel",
                           _excel_bytes(summary_frame),
                           "Bao_cao_thong_ke_LST.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with r2:
        try:
            pdf_bytes = _build_pdf(summary_frame, "BÁO CÁO KẾT QUẢ XỬ LÝ LST – HÀ NỘI")
            st.download_button("Tải báo cáo PDF", pdf_bytes,
                               "Bao_cao_LST_HaNoi.pdf", "application/pdf")
        except Exception as exc:
            st.warning(f"Không tạo được PDF: {exc}")
        try:
            docx_bytes = _build_docx(summary_frame, "BÁO CÁO KẾT QUẢ XỬ LÝ LST – HÀ NỘI")
            st.download_button("Tải báo cáo DOCX", docx_bytes,
                               "Bao_cao_LST_HaNoi.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        except Exception as exc:
            st.warning(f"Không tạo được DOCX: {exc}")


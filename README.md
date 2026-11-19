# Demo bản đồ nhiệt độ bề mặt Hà Nội (LST) – bản đã sửa

## Đã sửa
- Hotspot và bảng xếp hạng dùng chung mảng LST đã lọc QA ở bước xử lý chính, tránh tạo lại mặt nạ khác khiến bản đồ hotspot trắng.
- Xuất Hotspot GeoTIFF với NoData đúng cho các pixel không hợp lệ.
- Đọc thuộc tính GeoJSON bằng thư viện chuẩn `json`, không còn lỗi thiếu `geopandas`.
- Bảng xếp hạng xử lý an toàn khi không có ô lưới hợp lệ.

## Chạy trên Windows
Mở CMD tại thư mục chứa `app.py` và `requirements.txt`, chạy:

```bat
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Sau đó mở `http://localhost:8501`.

## Dữ liệu cần tải lên
Bắt buộc: `ST_B10.TIF` và `QA_PIXEL.TIF` cùng một cảnh Landsat 8/9 Collection 2 Level-2.

Tùy chọn: `MTL.txt`, `ST_QA.TIF`, `SR_B3.TIF`, `SR_B4.TIF`, `SR_B5.TIF`, `SR_B6.TIF`, raster LST tham chiếu USGS và ranh giới GeoJSON.

## Lưu ý
- Bảng xếp hạng hiện là theo ô lưới của ảnh xem trước, không phải theo quận/huyện.
- So sánh sai số USGS chỉ thực hiện khi raster tham chiếu cùng CRS và cùng lưới tọa độ với ST_B10.
- Các GeoTIFF được ghi rõ là ảnh xem trước giảm mẫu.

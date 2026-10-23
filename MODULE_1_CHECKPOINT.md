# Module 1 final boundary

Owner: **Vennela Laxmi Puli**

Module 1 ends when the system can:

1. select an AGIQA-3K image;
2. load the original prompt and MOS metadata;
3. detect one or more reliable regions, or create a `full_image` fallback;
4. expose region IDs, bounding boxes and segmentation data;
5. return the handoff endpoint:

```http
GET /api/v1/images/{image_id}/handoff
```

The handoff retains:

- `image_id`
- `dataset_image_name`
- original image URL
- original prompt
- generator
- AGIQA-3K quality/alignment MOS and STD
- region IDs
- labels
- confidence
- normalized bounding boxes
- segmentation polygons
- mask URLs when available

No caption model is required.

# MODULE 1 ENDS HERE

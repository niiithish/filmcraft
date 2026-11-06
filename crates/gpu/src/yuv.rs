//! Export output as planar BT.709 limited 8-bit 4:2:0, converted on the device (`yuv.wgsl`), for
//! encoders that take it (VA-API): only 1.5 bytes a pixel come back instead of the 8 of the
//! half-float accumulator, and the CPU does no colour conversion. The arithmetic is the CPU
//! path's (sRGB through the same table, then `rgba_to_yuv420_8`'s formula); GPU float division
//! may round the last bit differently, so a code can differ by one.

/// Workgroup size of `yuv.wgsl` in both dimensions.
const GROUP: u32 = 8;
/// Words of the lookup table: the sRGB codes, then four 256-entry `f32` tables.
const LUT_WORDS: usize = 4096 + 4 * 256;

pub(crate) struct YuvStage {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    lut: wgpu::Buffer,
    params: wgpu::Buffer,
    /// (storage output, mappable read-back copy, size in bytes) for the current frame size.
    out: Option<(wgpu::Buffer, wgpu::Buffer, u64)>,
}

/// Whether the stage converts a `w`×`h` picture: whole words per invocation (width a multiple of
/// 8, even height), and an output the device can bind.
pub(crate) fn takes(device: &wgpu::Device, w: u32, h: u32) -> bool {
    let len = yuv_len(w, h);
    w > 0 && h > 0 && w.is_multiple_of(8) && h.is_multiple_of(2) && len.is_some_and(|n| n <= device.limits().max_storage_buffer_binding_size)
}

/// Bytes of a planar 4:2:0 `w`×`h` picture (even sizes).
fn yuv_len(w: u32, h: u32) -> Option<u64> {
    let luma = u64::from(w).checked_mul(u64::from(h))?;
    luma.checked_add(luma / 2)
}

fn lut_bytes() -> Vec<u8> {
    let srgb = filmcraft_color::linear_to_srgb_u8_table();
    let unit: [f32; 256] = std::array::from_fn(|c| c as f32 / 255.0);
    let mut words: Vec<u32> = Vec::with_capacity(LUT_WORDS);
    words.extend(srgb.iter().map(|&c| u32::from(c)));
    words.extend(unit.iter().map(|u| u.to_bits()));
    for k in [0.2126f32, 0.7152, 0.0722] {
        words.extend(unit.iter().map(|u| (k * u).to_bits()));
    }
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

impl YuvStage {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("filmcraft-yuv"),
            source: wgpu::ShaderSource::Wgsl(include_str!("yuv.wgsl").into()),
        });
        let buffer = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yuv"),
            entries: &[
                buffer(0, wgpu::BufferBindingType::Uniform),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                buffer(2, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer(3, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("yuv"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yuv"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let table = lut_bytes();
        let lut = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yuv-lut"),
            size: table.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&lut, 0, &table);
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yuv-params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, bgl, lut, params, out: None }
    }

    /// Record the conversion of `src` (a `w`×`h` accumulator, see [`takes`]) and the copy of the
    /// result to the read-back buffer; the byte size of the picture, or `None` for a size the
    /// stage does not take.
    pub(crate) fn record(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        w: u32,
        h: u32,
    ) -> Option<u64> {
        if !takes(device, w, h) {
            return None;
        }
        let len = yuv_len(w, h)?;
        if self.out.as_ref().is_none_or(|(_, _, n)| *n != len) {
            let storage = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("yuv-out"),
                size: len,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("yuv-readback"),
                size: len,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            self.out = Some((storage, readback, len));
        }
        let (storage, readback, _) = self.out.as_ref()?;
        let params: Vec<u8> = [w, h, w / 2, h / 2].iter().flat_map(|v| v.to_le_bytes()).collect();
        queue.write_buffer(&self.params, 0, &params);
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yuv"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(src) },
                wgpu::BindGroupEntry { binding: 2, resource: self.lut.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: storage.as_entire_binding() },
            ],
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("yuv"), timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((w / 8).div_ceil(GROUP), (h / 2).div_ceil(GROUP), 1);
        }
        enc.copy_buffer_to_buffer(storage, 0, readback, 0, len);
        Some(len)
    }

    /// The picture [`Self::record`] converted, once the GPU has finished it.
    pub(crate) fn read(&self, device: &wgpu::Device, len: u64) -> Option<Vec<u8>> {
        let (_, readback, size) = self.out.as_ref()?;
        if *size != len {
            return None;
        }
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        crate::wait_mapped(device, &rx)?;
        let data = slice.get_mapped_range().ok()?;
        let out = data.to_vec();
        drop(data);
        readback.unmap();
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_layout() {
        let b = lut_bytes();
        assert_eq!(b.len(), LUT_WORDS * 4);
        let word = |i: usize| u32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
        assert_eq!(word(0), 0);
        assert_eq!(word(4095), 255);
        assert_eq!(f32::from_bits(word(4096 + 255)), 1.0);
        assert_eq!(f32::from_bits(word(4352 + 255)), 0.2126);
    }

    #[test]
    fn sizes() {
        assert_eq!(yuv_len(1080, 1920), Some(1080 * 1920 * 3 / 2));
        // a size whose planes don't fit in u64 is refused, not wrapped
        assert_eq!(yuv_len(u32::MAX, u32::MAX), None);
    }
}

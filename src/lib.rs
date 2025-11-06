mod tensor_batch;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::tensor_batch::AppendError;

pub struct BatchRingBuffer {
    buffer: Vec<tensor_batch::TensorBatch>,
    tail: AtomicUsize,
    head: AtomicUsize,
    // The mask is used for efficient modulo arithmetic to wrap around the ring buffer.
    // The Problem:
    // When you have a ring buffer with n buffers, you need to convert a continuously incrementing index (0, 1, 2, 3, 4, 5...) into a buffer position (0, 1, 2, 3, 0, 1, 2, 3...).
    // Normally you'd use: buffer_idx = head % buffer_count
    // The Optimization:
    // If buffer_count is a power of 2 (e.g., 4, 8, 16), you can replace the slow modulo operation with a fast bitwise AND:
    // If buffer_count = 4 (which is 2^2)
    // mask = 4 - 1 = 3 = 0b0011

    // head = 0  → 0 & 0b0011 = 0
    // head = 1  → 1 & 0b0011 = 1
    // head = 2  → 2 & 0b0011 = 2
    // head = 3  → 3 & 0b0011 = 3
    // head = 4  → 4 & 0b0011 = 0  // Wraps around!
    // head = 5  → 5 & 0b0011 = 1
    // head = 6  → 6 & 0b0011 = 2
    mask: usize,
}

impl BatchRingBuffer {
    pub fn new(batches_nr: usize, tensor_size: usize, capacity: usize) -> BatchRingBuffer {
        // TODO: check batches_nr is power of 2
        let mut buffer = Vec::with_capacity(batches_nr);
        for _ in 0..batches_nr {
            buffer.push(tensor_batch::TensorBatch::new(tensor_size, capacity));
        }
        BatchRingBuffer {
            buffer,
            tail: AtomicUsize::new(0),
            head: AtomicUsize::new(0),
            mask: batches_nr - 1,
        }
    }

    /// Append data to the current buffer, moving to next if full
    pub fn append(&self, data: &[u8]) -> Result<usize, AppendError> {
        loop {
            // head is where the current buffer is, as it is the hottest path.
            // taking an optimistic approach is preferable: try to append the batch
            // to this one and handle the failure with more atomics as needed.
            let head = self.head.load(Ordering::Acquire);

            // Wraps around like modulo, see the definition of self.mask for how it works
            let buffer_idx = head & self.mask;
            let buffer = &self.buffer[buffer_idx];

            // Try to append to current buffer
            match buffer.append(data) {
                Some(()) => return Ok(buffer_idx),
                None => {
                    // Buffer is full, try to move to next
                    let tail = self.tail.load(Ordering::Acquire);

                    // Check if we have space (at least one buffer available)
                    let used = head.wrapping_sub(tail);
                    if used >= self.buffer.len() {
                        return Err(AppendError::AllBuffersFull);
                    }

                    // Use CAS to ensure only one thread advances head
                    match self.head.compare_exchange_weak(
                        head,
                        head + 1,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => {
                            // Successfully moved to next buffer, retry append
                            continue;
                        }
                        Err(_) => {
                            // Another thread moved head, retry with new head
                            continue;
                        }
                    }
                }
            }
        }
    }

    pub fn get_ready_buffer() {}
}

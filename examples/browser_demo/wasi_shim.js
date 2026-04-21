// Minimal WASI shim for the Almide WASM Whisper module.
//
// transcribe() takes model_bytes and mel_bytes already in WASM memory, so
// the file-I/O imports (path_open, fd_read, etc.) only need to exist for
// link compatibility — they should never fire at runtime. fd_write is the
// only real one (for println/stderr), plus clock_time_get and random_get.

export function makeWasi({ stdout = console.log, stderr = console.error } = {}) {
  let memory = null;
  const setMemory = (m) => { memory = m; };

  const ERRNO_SUCCESS = 0;
  const ERRNO_BADF = 8;
  const ERRNO_NOSYS = 52;
  const ERRNO_NOENT = 44;

  function readString(ptr, len) {
    return new TextDecoder().decode(new Uint8Array(memory.buffer, ptr, len));
  }

  function fd_write(fd, iovs_ptr, iovs_len, nwritten_ptr) {
    const view = new DataView(memory.buffer);
    let total = 0;
    let chunks = [];
    for (let i = 0; i < iovs_len; i++) {
      const buf = view.getUint32(iovs_ptr + i * 8, true);
      const len = view.getUint32(iovs_ptr + i * 8 + 4, true);
      chunks.push(new Uint8Array(memory.buffer, buf, len));
      total += len;
    }
    const merged = new Uint8Array(total);
    let offset = 0;
    for (const c of chunks) { merged.set(c, offset); offset += c.length; }
    const text = new TextDecoder().decode(merged);
    if (fd === 1) stdout(text.replace(/\n$/, ''));
    else if (fd === 2) stderr(text.replace(/\n$/, ''));
    view.setUint32(nwritten_ptr, total, true);
    return ERRNO_SUCCESS;
  }

  function clock_time_get(_id, _precision, time_ptr) {
    const view = new DataView(memory.buffer);
    const ns = BigInt(Date.now()) * 1000000n;
    view.setBigUint64(time_ptr, ns, true);
    return ERRNO_SUCCESS;
  }

  function random_get(buf_ptr, buf_len) {
    const bytes = new Uint8Array(memory.buffer, buf_ptr, buf_len);
    crypto.getRandomValues(bytes);
    return ERRNO_SUCCESS;
  }

  function proc_exit(code) {
    throw new Error(`WASI proc_exit(${code})`);
  }

  // All file/path operations: return BADF (no real fs in browser).
  const noFs = () => ERRNO_BADF;
  const enoent = () => ERRNO_NOENT;

  // environ_*, args_* return empty.
  function environ_sizes_get(count_ptr, size_ptr) {
    const v = new DataView(memory.buffer);
    v.setUint32(count_ptr, 0, true);
    v.setUint32(size_ptr, 0, true);
    return ERRNO_SUCCESS;
  }
  function args_sizes_get(count_ptr, size_ptr) {
    const v = new DataView(memory.buffer);
    v.setUint32(count_ptr, 0, true);
    v.setUint32(size_ptr, 0, true);
    return ERRNO_SUCCESS;
  }
  const empty_get = () => ERRNO_SUCCESS;

  return {
    setMemory,
    imports: {
      wasi_snapshot_preview1: {
        fd_write,
        fd_read: noFs,
        fd_close: noFs,
        fd_seek: noFs,
        fd_filestat_get: noFs,
        fd_fdstat_get: noFs,
        fd_fdstat_set_flags: noFs,
        fd_prestat_get: enoent,
        fd_prestat_dir_name: noFs,
        path_open: enoent,
        path_filestat_get: enoent,
        path_create_directory: noFs,
        path_rename: noFs,
        path_unlink_file: noFs,
        path_remove_directory: noFs,
        path_symlink: noFs,
        path_link: noFs,
        path_readlink: noFs,
        fd_readdir: noFs,
        fd_renumber: noFs,
        fd_sync: noFs,
        fd_datasync: noFs,
        fd_advise: noFs,
        fd_allocate: noFs,
        fd_pread: noFs,
        fd_pwrite: noFs,
        fd_filestat_set_size: noFs,
        fd_filestat_set_times: noFs,
        fd_tell: noFs,
        path_filestat_set_times: noFs,
        clock_res_get: () => ERRNO_SUCCESS,
        clock_time_get,
        random_get,
        proc_exit,
        environ_sizes_get,
        environ_get: empty_get,
        args_sizes_get,
        args_get: empty_get,
        poll_oneoff: () => ERRNO_NOSYS,
        sched_yield: () => ERRNO_SUCCESS,
      },
    },
  };
}

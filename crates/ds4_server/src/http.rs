//! Minimal HTTP/1.1 + SSE over `std::net` — no async runtime, no framework
//! (the workspace vendors neither). Mirrors how the antirez ds4-server serves
//! raw sockets. Single connection at a time (the GPU session is single-threaded).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

const CORS: &str = "Access-Control-Allow-Origin: *\r\n\
Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
Access-Control-Allow-Headers: *\r\n";

/// Read one HTTP request (request line + headers + Content-Length body).
/// Returns `Ok(None)` on a cleanly closed connection.
pub fn read_request(stream: &TcpStream) -> std::io::Result<Option<Request>> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();

    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 {
            break;
        }
        let t = h.trim_end();
        if t.is_empty() {
            break;
        }
        if let Some((k, v)) = t.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    Ok(Some(Request { method, path, body }))
}

fn status_text(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    }
}

/// Write a buffered JSON response.
pub fn write_json(stream: &mut TcpStream, status: u16, body: &str) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{CORS}Connection: close\r\n\r\n",
        status,
        status_text(status),
        body.len()
    )?;
    stream.write_all(body.as_bytes())?;
    stream.flush()
}

/// 204 for CORS preflight (OPTIONS).
pub fn write_no_content(stream: &mut TcpStream) -> std::io::Result<()> {
    write!(stream, "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n{CORS}Connection: close\r\n\r\n")?;
    stream.flush()
}

/// Begin an SSE response (`text/event-stream`); caller then streams events.
pub fn write_sse_headers(stream: &mut TcpStream) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n{CORS}Connection: close\r\n\r\n"
    )?;
    stream.flush()
}

/// Write one SSE `data:` event (and flush so the client sees it immediately).
pub fn write_sse_data(stream: &mut TcpStream, data: &str) -> std::io::Result<()> {
    write!(stream, "data: {data}\n\n")?;
    stream.flush()
}

/// Write an SSE event with an explicit `event:` type (Anthropic protocol).
pub fn write_sse_event(stream: &mut TcpStream, event: &str, data: &str) -> std::io::Result<()> {
    write!(stream, "event: {event}\ndata: {data}\n\n")?;
    stream.flush()
}

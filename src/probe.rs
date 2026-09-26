//! Answer a pre-`initialize` `server/discover` probe before rmcp sees it.
//!
//! Claude Code 2.1.282 (2026-09-25) opens every connection with a
//! `server/discover` probe for protocol 2026-07-28. We do not speak that
//! revision (see `supported_protocol_versions`), so the probe is refused and
//! the client falls back to `initialize` for 2025-11-25. That part works.
//!
//! What broke is rmcp 3.1.0 (still so in 3.4.1): when the first message of a
//! session is not `initialize`, it marks the session "every request carries a
//! 2026-07-28 `_meta` envelope" *before* handling the request, and never clears
//! the mark -- not when it refuses the probe, not when `initialize` follows.
//! The fallback session then rejects the client's plain `tools/list` with
//! -32602 "request _meta is missing", and the client shows no tools at all.
//!
//! So a probe for a version we do not support is answered here, with the same
//! -32022 error rmcp would send, and never reaches rmcp. rmcp's first message
//! is then `initialize`, and the session is an ordinary 2025-11-25 one. Any
//! other opening (no version, or one we support) passes through untouched.
//!
//! Remove this once rmcp stops latching on a refused probe, or once the server
//! implements 2026-07-28.

use std::{borrow::Cow, future::Future};

use rmcp::{
    model::{
        ClientRequest, ErrorData, GetMeta, JsonRpcMessage, ProtocolVersion, ServerJsonRpcMessage,
    },
    service::{RxJsonRpcMessage, TxJsonRpcMessage},
    transport::Transport,
    RoleServer,
};

pub struct DiscoverProbeGuard<T> {
    inner: T,
    supported: Cow<'static, [ProtocolVersion]>,
    /// Set once `initialize` has gone through; from then on this is a no-op.
    initialized: bool,
}

impl<T> DiscoverProbeGuard<T> {
    pub fn new(inner: T, supported: Cow<'static, [ProtocolVersion]>) -> Self {
        Self {
            inner,
            supported,
            initialized: false,
        }
    }
}

impl<T> Transport<RoleServer> for DiscoverProbeGuard<T>
where
    T: Transport<RoleServer>,
{
    type Error = T::Error;

    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleServer>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        self.inner.send(item)
    }

    async fn receive(&mut self) -> Option<RxJsonRpcMessage<RoleServer>> {
        loop {
            let msg = self.inner.receive().await?;
            if self.initialized {
                return Some(msg);
            }
            if let JsonRpcMessage::Request(req) = &msg {
                match &req.request {
                    ClientRequest::InitializeRequest(_) => self.initialized = true,
                    ClientRequest::DiscoverRequest(_) => {
                        if let Some(version) = req.request.get_meta().protocol_version() {
                            if !self.supported.contains(&version) {
                                let error = ErrorData::unsupported_protocol_version(
                                    version,
                                    &self.supported,
                                );
                                let reply =
                                    ServerJsonRpcMessage::error(error, Some(req.id.clone()));
                                if self.inner.send(reply).await.is_err() {
                                    return None;
                                }
                                continue;
                            }
                        }
                    }
                    _ => {}
                }
            }
            return Some(msg);
        }
    }

    fn close(&mut self) -> impl Future<Output = Result<(), Self::Error>> + Send {
        self.inner.close()
    }
}

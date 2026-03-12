//! Handler dispatch — routes tool names to handler functions.
//!
//! Each handler is in its own module file to keep CC ≤ 7.

use anyhow::Result;
use serde_json::Value;
use sqlx::PgPool;

pub mod alma;
pub mod alarma;
pub mod cronica;
pub mod decreto;
pub mod eco;
pub mod expediente;
pub mod faro;
pub mod jornada;
pub mod puente;
pub mod remedio;
pub mod vigia;
pub mod zafra;

/// Dispatch a tool call to the appropriate handler.
///
/// Returns MCP content response (text or structured).
pub async fn dispatch(pool: &PgPool, tool_name: &str, args: Value) -> Result<Value> {
    let result = match tool_name {
        "cuba_alma" => alma::handle(pool, args).await?,
        "cuba_cronica" => cronica::handle(pool, args).await?,
        "cuba_faro" => faro::handle(pool, args).await?,
        "cuba_puente" => puente::handle(pool, args).await?,
        "cuba_eco" => eco::handle(pool, args).await?,
        "cuba_alarma" => alarma::handle(pool, args).await?,
        "cuba_remedio" => remedio::handle(pool, args).await?,
        "cuba_expediente" => expediente::handle(pool, args).await?,
        "cuba_jornada" => jornada::handle(pool, args).await?,
        "cuba_decreto" => decreto::handle(pool, args).await?,
        "cuba_vigia" => vigia::handle(pool, args).await?,
        "cuba_zafra" => zafra::handle(pool, args).await?,
        _ => {
            tracing::warn!(tool = %tool_name, "unknown tool");
            anyhow::bail!("Unknown tool: {tool_name}")
        }
    };

    // Wrap in MCP content format
    Ok(serde_json::json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string(&result)?
        }]
    }))
}

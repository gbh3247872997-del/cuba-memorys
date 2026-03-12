//! Handler: cuba_expediente — Search past errors/solutions.

use anyhow::Result;
use serde_json::Value;
use sqlx::PgPool;

pub async fn handle(pool: &PgPool, args: Value) -> Result<Value> {
    let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
    let project = args.get("project").and_then(|v| v.as_str());
    let resolved_only = args.get("resolved_only").and_then(|v| v.as_bool()).unwrap_or(false);
    let proposed_action = args.get("proposed_action").and_then(|v| v.as_str());

    if query.is_empty() {
        anyhow::bail!("query is required");
    }

    // Search errors
    let mut sql = String::from(
        "SELECT id, error_type, error_message, solution, resolved, project,
                similarity(error_message, $1)::float8 AS sim
         FROM brain_errors
         WHERE (search_vector @@ plainto_tsquery('simple', $1) OR similarity(error_message, $1) > 0.3)"
    );

    if resolved_only { sql.push_str(" AND resolved = true"); }
    if let Some(p) = project { sql.push_str(&format!(" AND project = '{}'", p.replace('\'', "''"))); }
    sql.push_str(" ORDER BY sim DESC LIMIT 20");

    let errors: Vec<(uuid::Uuid, String, String, Option<String>, bool, String, f64)> =
        sqlx::query_as(&sql).bind(query).fetch_all(pool).await?;

    let results: Vec<Value> = errors.iter().map(|(id, et, em, sol, res, proj, sim)| {
        serde_json::json!({
            "id": id.to_string(), "error_type": et, "error_message": &em[..em.len().min(200)],
            "solution": sol, "resolved": res, "project": proj, "similarity": sim
        })
    }).collect();

    // Anti-repetition guard
    let mut response = serde_json::json!({"query": query, "results": results, "count": results.len()});

    if let Some(action) = proposed_action {
        let failed_similar: Vec<(String,)> = sqlx::query_as(
            "SELECT error_message FROM brain_errors
             WHERE resolved = false AND similarity(solution, $1) > 0.5 LIMIT 3"
        )
        .bind(action)
        .fetch_all(pool)
        .await?;

        if !failed_similar.is_empty() {
            response["anti_repetition_warning"] = serde_json::json!(
                format!("⚠️ Similar approach failed {} time(s) before", failed_similar.len())
            );
        }
    }

    Ok(response)
}

//! The host end of a device, named as `kind:argument` (`tcp:8880`, `unix:/run/x`, `log:a.log`). Each device type's `backend`
//! key reads its strings through here and decides what its kinds mean.

use serde::{Deserialize, Deserializer};

/// `kind:argument`, split at the first colon so an argument may hold more (`log:C:\x.log`).
pub fn split(s: &str) -> Result<(&str, &str), String> {
    s.split_once(':').filter(|(k, a)| !k.is_empty() && !a.is_empty()).ok_or_else(|| format!("backend \"{s}\" is not KIND:ARGUMENT"))
}

/// A `backend` value: one string, or a list of them.
pub fn list<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Backends {
        One(String),
        Many(Vec<String>),
    }
    Ok(match Backends::deserialize(d)? {
        Backends::One(s) => vec![s],
        Backends::Many(v) => v,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_backend_splits_at_the_first_colon() {
        assert_eq!(split("tcp:8880"), Ok(("tcp", "8880")));
        assert_eq!(split("log:C:\\x.log"), Ok(("log", "C:\\x.log")));
        for bad in ["tcp", "tcp:", ":8880", ""] {
            assert!(split(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn one_string_or_a_list_reads_as_a_list() {
        #[derive(Deserialize)]
        struct T(#[serde(deserialize_with = "list")] Vec<String>);
        let one: T = serde::Deserialize::deserialize(toml::Value::String("tcp:1".into())).unwrap();
        let many: T = serde::Deserialize::deserialize(toml::Value::Array(vec!["a:1".into(), "b:2".into()])).unwrap();
        assert_eq!((one.0, many.0), (vec!["tcp:1".to_string()], vec!["a:1".to_string(), "b:2".to_string()]));
    }
}

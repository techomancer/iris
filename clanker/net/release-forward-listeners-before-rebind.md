# Release old forwarding sockets before rebinding their ports

Editing live NAT forwards could log `Address already in use` and lose rules
whose host ports appeared in both the old and new sets. Changing the guest
port or switching between loopback and all-interface binds exposed it.

`NatEngine::rebind_forwards` split off transient FTP listeners but left the
static TCP listeners and UDP sockets alive during `bind_forwards`. Assigning
the replacement vectors released the old sockets only after binding failed.

Clear both static listener collections before binding replacements. Preserve
the transient FTP listeners and established connection maps. The regression
`live_forward_rebind_reuses_ports_and_preserves_transient_listeners` covers
reused TCP/UDP ports, bind scopes, removal and the transient listener. Its FTP
listener must be nonblocking, as the real FTP path is: a pump polls `accept`.

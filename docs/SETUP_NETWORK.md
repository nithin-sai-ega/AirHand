# Network and phone-app setup

The phone app pushes sensor data to the laptop over Wi-Fi. For that to work:

1. The phone and laptop must be on the same network, and the network must allow devices to
   talk to each other.
2. Windows Firewall must allow inbound traffic on the probe's ports.
3. The phone app must be pointed at the laptop's IP address and port.

Default ports (from `configs/probe.yaml`): **HTTP 8000/TCP** and **UDP 5005**.

## 1. Choose a network

| Situation | What to do |
|---|---|
| Home Wi-Fi | Connect both devices to the same Wi-Fi. Usually works as-is. |
| Campus / office Wi-Fi | These often enable *client isolation* (devices can't reach each other). If the probe receives nothing, use a hotspot instead. |
| iPhone Personal Hotspot | Turn on the hotspot on the iPhone and connect the laptop to it. The laptop usually gets an address like `172.20.10.x`. |
| Windows Mobile Hotspot | Settings -> Network & internet -> Mobile hotspot. Connect the iPhone to it. The laptop is usually `192.168.137.1`. |

## 2. Find the laptop's IP address

The probe prints its URLs on startup. You can also run:

```powershell
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" } | Select-Object InterfaceAlias, IPAddress
```

Use the address of the Wi-Fi (or hotspot) adapter.

## 3. Allow the ports in Windows Firewall

Check whether Windows treats the network as **Private** or **Public**:

```powershell
Get-NetConnectionProfile | Select-Object Name, InterfaceAlias, NetworkCategory
```

On your home Wi-Fi it should be **Private**. If it shows Public, change it in
Settings -> Network & internet -> Wi-Fi -> (your network) -> Network profile type -> Private.
Only do this for networks you trust.

Then, in a PowerShell window opened with **Run as administrator**, add rules that apply to
Private networks only:

```powershell
New-NetFirewallRule -DisplayName "AirHand probe HTTP" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private
```

```powershell
New-NetFirewallRule -DisplayName "AirHand probe UDP" -Direction Inbound -Protocol UDP -LocalPort 5005 -Action Allow -Profile Private
```

The first time the probe runs, Windows may also show a "Windows Security Alert" for
`python.exe`. Allow it on **Private networks** only.

To remove the rules later:

```powershell
Remove-NetFirewallRule -DisplayName "AirHand probe*"
```

### Quick connectivity check (before involving the watch)

With the probe running, open `http://<laptop-ip>:8000/test` in Safari on the iPhone. The page
should show `ok`, and the probe should print a `GET /test` line. If it doesn't:
- check both devices are on the same network,
- check the firewall rule and network profile,
- try a hotspot (client isolation).

## 4. Phone app: Sensor Logger (primary)

Sensor Logger (by Choi Tsz Hei) is free on the App Store and has an Apple Watch companion app.
The exact menu names below may differ between app versions; follow what the app shows.

1. Install Sensor Logger on the iPhone and make sure its Watch app is installed on the watch
   (Watch app on iPhone -> Available apps).
2. In Sensor Logger's settings, find the **data streaming / HTTP push** option.
   - Push URL: `http://<laptop-ip>:8000/data`
   - Enable it. If there is a batch / push period setting, note its value; it directly
     affects latency and will be recorded in the session note.
3. Enable the **Apple Watch accelerometer and gyroscope** (and the watch's orientation /
   device-motion output if offered). Set the highest sampling rate available and note it.
4. Start a recording. Packets should start appearing in the probe console.

**Feasibility question this answers:** whether *watch* sensor data is streamed live through
the phone, or only recorded and transferred afterwards. If only phone sensors arrive live,
stop and report back before going further.

## 5. Phone app: SensorLog (backup, UDP)

If Sensor Logger cannot stream watch data live, try SensorLog with UDP streaming to
`<laptop-ip>` port `5005`. Run the probe as usual; it listens on both ports at once.

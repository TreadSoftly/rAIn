<div align="center">

# ~ Argos ~

</div>

<div align="center">

### Install → Run (Quickstart)

<table align="center" cellpadding="6" style="border-collapse:collapse; text-align:left;">
  <tr><td>•</td><td><b>Clone</b> → open a terminal in the repo root.</td></tr>
  <tr><td>•</td><td><b>Zero-touch bootstrap</b>: first run auto-creates a venv, installs Torch (CPU if no CUDA), installs the project editable, fetches chosen weights, and writes launchers.</td></tr>
  <tr><td>•</td><td><b>Media in</b> → put images/videos in <code>projects/argos/tests/raw/</code>. (Assets live in <code>tests/assets</code>.)</td></tr>
  <tr><td>•</td><td><b>Results out</b> → processed files land in <code>projects/argos/tests/results/</code> with helpful suffixes (e.g., <code>_boxes</code>, <code>_heat</code>).</td></tr>
</table>

</div>

<div align="center">

## To Get Up And Running

</div>

<div align="center">

> In Your Terminal Copy And Paste Commands Below Into Terminal: >

</div>

```
git clone https://github.com/TreadSoftly/rAIn.git

cd rAIn

```

> Powershell

```
.\build

```

> Bash

```
./build

```

###### Once you run the .\build or ./build you can just rerun a regular command "build" or "build argos" or "argos build" to get the build selection process again

<br>

<div align="center">

> Help Commands: >

</div>

```
argos -h

```

> or

```
argos ?

```

<br>

<div align="center">

> To practice with the media in the tests/raw folder copy and paste: >

</div>

```
d mildrone

```

```
assets heatmap

```

```
hm all

```

```
all detect

```

<div align="center">

> IMPORTANT: = To Process Your Own Media Files Locate /tests/raw/ Folder And Place Your Media There. Then Run The Desired Commands As Shown Above To Process The Media. Results Will Be Placed In /tests/results/ Folder.

</div>

<div align="center">

<table style="display:inline-block; text-align:left; border-collapse:collapse;">
<tr><td style="padding:0 .5em;">•</td><td>Troubleshooting steps and adjustments to come. This project is a continuous one man work in progress, it is what it is.</td></tr>

</table>

</div>

<br>

<div align="center">

# ~ Capabilities Demo Videos ~

<div align="center">

<table style="display:inline-block; text-align:left; border-collapse:collapse;">
<tr><td style="padding:0 .5em;">•</td><td>Tracking</td></tr>
<tr><td style="padding:0 .5em;">•</td><td>Detection + ID Boxes + Confidence Percentage</td></tr>
<tr><td style="padding:0 .5em;">•</td><td>Segmented Multi-Color + ID With/and/Without Boxes + Confidence Percentage</td></tr>
<tr><td style="padding:0 .5em;">•</td><td>Only Segmented Highlights – No ID / No Confidence Percentage</td></tr>

</table>

</div>

<table align="center">

<tr>

<td align="center" valign="top">

<strong>
Multi-Color Segmentation Highlights + Tracking + ID + Confidence Percentages
</strong><br>
<video src="https://github.com/user-attachments/assets/d9b6f0f8-6ea2-4f29-813e-96ff84ae1ab2" width="500" controls></video>

</td>

<td align="center" valign="top">

<strong>
Same-Color Segmented Highlights + ID Boxes + Tracking + Confidence Percentages:
</strong> <em>✨ALSO MUSIC✨</em><br>
<video src="https://github.com/user-attachments/assets/07168b7d-7589-4948-bbdd-55f4c723d33d" width="500" controls></video>

</td>

</tr>

<tr>

<td align="center" valign="top">

<strong>
Detection + ID Boxes + Tracking + Confidence Percentages | -No Segmented Highlights
</strong><br>
<video src="https://github.com/user-attachments/assets/a6a42d82-faed-4c91-912a-81a3e8da27ec" width="500" controls></video>

</td>

<td align="center" valign="top">

<strong>
Segmented Highlights + Tracking | -No ID / -No Confidence Percentages
</strong><br>
<video src="https://github.com/user-attachments/assets/5ed5b57c-8c9b-4578-aee7-9ed80c9ff8e5" width="500" controls></video>

</td>

</tr>

</table>

<br>

<div align="center">

# ~ Capabilities Demo Images ~

<div align="center">

<table style="display:inline-block; text-align:left; border-collapse:collapse;">
<tr><td style="padding:0 .5em;">•</td><td>Detection ID Boxes + Confidence Percentage</td></tr>
<tr><td style="padding:0 .5em;">•</td><td>Segmented Multi-Color Highlights + ID + Confidence Percentage</td></tr>
</table>

</div>

<table align="center" cellpadding="5">

<!-- Row 1 -->
<tr>
<td align="center"><img src="https://github.com/user-attachments/assets/db45bc31-3b7d-4001-b868-eb37589de66c" width="300" alt="Image"></td>
<td align="center"><img src="https://github.com/user-attachments/assets/84fac12c-2ddd-4d2e-8f5c-3a92dd8661b7" width="300" alt="Image"></td>
<td align="center"><img src="https://github.com/user-attachments/assets/c8f90d41-beab-4ac8-ae62-df0c4814c3d6" width="300" alt="Image"></td>
<td align="center"><img src="https://github.com/user-attachments/assets/c2957623-577e-4cf6-b772-038fc6292a2a" width="300" alt="Image"></td>
</tr>

<!-- Row 2 -->
<tr>
<td align="center"><img src="https://github.com/user-attachments/assets/f73edeb5-59c3-4e39-9cd9-f6f5b57a41bd" width="300" alt="Image"></td>
<td align="center"><img src="https://github.com/user-attachments/assets/3ce4e76b-e196-487a-b508-db475ee9140d" width="300" alt="Image"></td>
<td align="center"><img src="https://github.com/user-attachments/assets/5a62471c-9204-4f3f-bdda-b751f011528a" width="300" alt="Image"></td>
<td align="center"><img src="https://github.com/user-attachments/assets/ea9e269f-3413-4469-8c4a-e79eaec13953" width="300" alt="Image"></td>
</tr>

<!-- Row 3 -->
<tr>
<td align="center"><img src="https://github.com/user-attachments/assets/274164d5-7673-4791-9953-96d3191ce615" width="300" alt="Image"></td>
<td align="center"><img src="https://github.com/user-attachments/assets/3742682a-6631-43bd-8de3-927b5f3f7221" width="300" alt="Image"></td>
<td align="center"><img src="https://github.com/user-attachments/assets/61c22df5-99ee-469c-9ebd-6f8dbf11a010" width="300" alt="Image"></td>
<td align="center"><img src="https://github.com/user-attachments/assets/60729b9f-65cb-4d70-b00d-c4a92807f9c2" width="300" alt="Image"></td>
</tr>

<!-- Row 4 -->
<tr>
<td align="center"><img src="https://github.com/user-attachments/assets/0891cd4d-d2a6-4222-a495-e3d098e01254" width="300" alt="Image"></td>
<td align="center"><img src="https://github.com/user-attachments/assets/3efba8f9-8dcb-4b6c-93ff-61c20c6c2d36" width="300" alt="Image"></td>
<td align="center"><img src="https://github.com/user-attachments/assets/27d90b52-252f-47ba-aa85-e775a359a435" width="300" alt="Image"></td>
<td align="center"><img src="https://github.com/user-attachments/assets/a6d52775-0134-4a8e-9d11-474518d09c89" width="300" alt="Image"></td>
</tr>

<!-- Row 5 (unchanged) -->
<tr>
<td align="center"><img src="https://github.com/user-attachments/assets/b09a82d2-f930-4204-a732-e437234e551d" width="300" alt="Image"></td>
<td align="center"><img src="https://github.com/user-attachments/assets/fc02e090-3869-4421-be44-6ae5337b109c" width="300" alt="Image"></td>
<td align="center"><img src="https://github.com/user-attachments/assets/d1d7a3d8-25a4-48e4-a6a3-4834bfbe324e" width="300" alt="Image"></td>
<td align="center"><img src="https://github.com/user-attachments/assets/9a48d1c7-0c7a-4ff6-8d22-a776cd8afe08" width="300" alt="Image"></td>
</tr>

</table> <!-- closes the main Image Gallery table -->

<br>

<div align="center">

# ~ Capabilities Map View Demo Images ~

<div align="center">

<table style="display:inline-block; text-align:left; border-collapse:collapse;">
<tr><td style="padding:0 .5em;">•</td><td>Geotagged GeoJson Files</td></tr>
</table>

</div>

<table align="center" cellpadding="5" cellspacing="0" border="0">
  <colgroup>
    <col width="300">
    <col width="300">
  </colgroup>

  <tr height="300">
    <td align="center" width="300" height="300" valign="middle">
      <video src="https://github.com/user-attachments/assets/c396456b-370b-485b-a5bc-b5f3852b40f3"
             width="300" controls playsinline muted></video>
    </td>
    <td align="center" width="300" height="300" valign="middle">
      <img src="https://github.com/user-attachments/assets/b406b50d-ea6a-4633-b716-e68b298029ec"
           width="300" height="300" alt="Geo Map 2">
    </td>
  </tr>

  <tr height="300">
    <td align="center" width="300" height="300" valign="middle">
      <img src="https://github.com/user-attachments/assets/d2613e96-7df7-4a1c-baf8-fa8680cfa57a"
           width="300" height="300" alt="Geo Map 3">
    </td>
    <td align="center" width="300" height="300" valign="middle">
      <img src="https://github.com/user-attachments/assets/24e0bc9e-b32a-48a5-813d-51bed1199f9b"
           width="300" height="300" alt="Geo Map 4">
    </td>
  </tr>
</table>

</div>
